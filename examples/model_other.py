"""
# TODO: needs implementing or not?
Calculate the normalising constant using CMCD."""
import jax
from absl import app, flags
from ml_collections.config_flags import config_flags
import wandb
import pickle
import inference_gym.using_jax as gym
# import models.logistic_regression as model_lr
# import models.seeds as model_seeds
import jax.numpy as jnp
from typing import Optional
from cmcd.run_lib import (
  training,
  sample_from_target,
  setup_training,
  )
from diffusionjax.utils import flatten_nested_dict

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", './configs/nice.py', "Training configuration.",
  lock_config=True)
flags.DEFINE_string("workdir", './examples/', "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

models_other = ["log_sonar", "log_ionosphere", "seeds"]


def load_model(model="log_sonar"):
    if model == "log_sonar":
        model, model_args = model_lr.load_model("sonar")
    if model == "log_ionosphere":
        model, model_args = model_lr.load_model("ionosphere")
    if model == "seeds":
        model, model_args = model_seeds.load_model()

    rng_key = jax.random.PRNGKey(1)
    (
        model_param_info,
        potential_fn,
        constrain_fn,
        _,
    ) = numpyro.infer.util.initialize_model(rng_key, model, model_args=model_args)
    params_flat, unflattener = ravel_pytree(model_param_info[0])
    log_prob_model = lambda z: -1.0 * potential_fn(unflattener(z))
    dim = params_flat.shape[0]
    unflatten_and_constrain = lambda z: constrain_fn(unflattener(z))
    return log_prob_model, dim, model


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  # num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

  # jax_config.update("jax_traceback_filtering", "off")
  # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

  log_prob_model, dim, model = load_model()
  sample_from_target_fn = None
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
    params, samples, target_samples, n_samples = training(
      config, log_prob_model, sample_from_target_fn, sample_shape)

    if config.wandb.log_artifact:
      artifact_name = f"{model}_{config.solver.outer_solver}_{config.solver.num_outer_steps}"
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
