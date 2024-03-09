"""Calculate the normalising constant using CMCD."""
import os
import pickle
from functools import partial
from diffusionjax.utils import flatten_nested_dict
from cmcd.utils import update_config_dict
from cmcd.run_lib import (
  training,
  sample_from_target,
  initialize,
  compute_bound,
  setup_training,
  get_solver,
  )
from ml_collections.config_flags import config_flags
from absl import app, flags
import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import multivariate_normal, norm
import wandb


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", './configs/funnel.py', "Training configuration.",
  lock_config=True)
flags.DEFINE_string("workdir", './examples/', "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


FUNNEL_EPS_DICT = {
    8: {"init_eps": 0.1, "lr": 0.01},
    16: {"init_eps": 0.1, "lr": 0.01},
    32: {"init_eps": 0.1, "lr": 0.005},
    64: {"init_eps": 0.1, "lr": 0.001},
    128: {"init_eps": 0.01, "lr": 0.01},
    256: {"init_eps": 0.01, "lr": 0.005},
}


def load_model(config):
  d = config.data.funnel_d
  sig = config.data.funnel_sig
  clip_y = config.data.funnel_clipy

  def neg_energy(x):
    def unbatched(x):
      v = x[0]
      log_density_v = norm.logpdf(v, loc=0.0, scale=3.0)
      variance_other = jnp.exp(v)
      other_dim = d - 1
      cov_other = jnp.eye(other_dim) * variance_other
      mean_other = jnp.zeros(other_dim)
      log_density_other = multivariate_normal.logpdf(
          x[1:], mean=mean_other, cov=cov_other
      )
      return log_density_v + log_density_other

    output = jnp.squeeze(jax.vmap(unbatched)(x[None, :]))
    return output

  def sample_data(rng, n_samples):
    # sample from Nd funnel distribution

    y_rng, x_rng = random.split(rng)

    y = (sig * random.normal(y_rng, (n_samples, 1))).clip(-clip_y, clip_y)
    x = random.normal(x_rng, (n_samples, d - 1)) * jnp.exp(-y / 2)
    return jnp.concatenate((y, x), axis=1)

  return neg_energy, d, sample_data


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
  sample_shape = (dim,)

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
    # Load in the correct LR from sweeps
    values = FUNNEL_EPS_DICT[config.solver.num_outer_steps]
    new_vals = {"init_eps": values["init_eps"], "lr": values["lr"]}
    update_config_dict(config, run, new_vals)
    params, samples, target_samples, n_samples = training(
      config, log_prob_model, sample_from_target_fn, sample_shape)
    sample_from_target(config, sample_from_target_fn, samples, target_samples, n_samples)

    if config.wandb.log_artifact:
      artifact_name = f"funnel_{config.solver.bound_mode}_{config.solver.num_outer_steps}"
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
