"""Calculate the normalising constant using CMCD."""
import os
import pickle
from diffusionjax.utils import flatten_nested_dict
import distrax
from cmcd.run_lib import (
  training,
  sample_from_target,
  setup_training,
  )
from ml_collections.config_flags import config_flags
from absl import app, flags
import jax
import jax.numpy as jnp
import jax.random as random
import wandb


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", './configs/many_gmm.py', "Training configuration.",
  lock_config=True)
flags.DEFINE_string("workdir", './examples/', "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


def load_model(config):
  gmm = GMM(dim=2, num_mixes=config.data.num_mixes, loc_scaling=config.data.loc_scaling)
  return gmm.log_prob, 2, gmm.sample


class GMM:
  def __init__(self, dim, num_mixes, loc_scaling, log_var_scaling=0.1, seed=0):
    self.seed = seed
    self.num_mixes = num_mixes
    self.dim = dim
    key = random.PRNGKey(seed)
    logits = jnp.ones(num_mixes)
    mean = (
        random.uniform(shape=(num_mixes, dim), key=key, minval=-1.0, maxval=1.0)
        * loc_scaling
    )
    log_var = jnp.ones(shape=(num_mixes, dim)) * log_var_scaling

    mixture_dist = distrax.Categorical(logits=logits)
    var = jax.nn.softplus(log_var)
    components_dist = distrax.Independent(
        distrax.Normal(loc=mean, scale=var), reinterpreted_batch_ndims=1
    )
    self.distribution = distrax.MixtureSameFamily(
        mixture_distribution=mixture_dist,
        components_distribution=components_dist,
    )

  def log_prob(self, x):
    log_prob = self.distribution.log_prob(x)

    # Can have numerical instabilities once log prob is very small. Manually override to prevent this.
    # This will cause the flow will ignore regions with less than 1e-4 probability under the target.
    valid_log_prob = log_prob > -1e4
    log_prob = jnp.where(valid_log_prob, log_prob, -jnp.inf * jnp.ones_like(log_prob))
    return log_prob

  def sample(self, seed, sample_shape):
    return self.distribution.sample(seed=seed, sample_shape=sample_shape)


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

  # jax_config.update("jax_traceback_filtering", "off")
  # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

  log_prob_model, dim, sample_from_target_fn = load_model(config)
  sample_shape = (dim, )

  wandb_kwargs = {
    "project": config.wandb.project,
    "entity": config.wandb.entity,
    "config": flatten_nested_dict(config.to_dict()),
    "name": config.wandb.name if config.wandb.name else None,
    "mode": "online" if config.wandb.log else "disabled",
    "settings": wandb.Settings(code_dir=config.wandb.code_dir),
  }

  with wandb.init(**wandb_kwargs) as run:
    setup_training(run)
    params = training(config, log_prob_model, sample_from_target_fn, sample_shape)

    if config.wandb.log_artifact:
      artifact_name = f"many_gmm_{config.solver.bound_mode}_{config.solver.num_outer_steps}"
      artifact = wandb.Artifact(
        artifact_name,
        type="final params",
      )
      # Save model
      with artifact.new_file("params.pkl", "wb") as f:
        pickle.dump(params, f)

      wandb.log_artifact(artifact)


if __name__ == "__main__":
    # os.environ["WANDB_API_KEY"] = "9835d6db89010f73306f92bb9a080c9751b25d28"

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    app.run(main)

