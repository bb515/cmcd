import os
import pickle
from functools import partial
import jax
import jax.numpy as jnp
import mcdboundingmachine as mcdbm
import ml_collections.config_flags
import numpy as np
import opt
import wandb
from absl import app, flags
from configs.base import TRACTABLE_DISTS
from jax.config import config as jax_config
from model_handler import load_model
from utils import (
  initialize,
  compute_ratio,
  compute_bound,
  calculate_W2_distances,
  flatten_nested_dict,
  log_final_losses,
  make_grid,
  )


def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
  config_dict.unlock()
  config_dict.update_from_flattened_dict(run.config)
  config_dict.update_from_flattened_dict(new_vals)
  run.config.update(new_vals, allow_val_change=True)
  config_dict.lock()


def setup_config(wandb_config, config):
  try:
    if wandb_config.model == "nice":
      config.model = (
        wandb_config.model
        + "_{}_{}_{}".format(wandb_config.alpha, wandb_config.n_bits, wandb_config.im_size))
      new_vals = {}
    elif wandb_config.model in ["funnel"]:
      pass
  except KeyError:
    new_vals = {}
    print("LR not found for model {} and boundmode {}".format(
      wand_config.model, wandb_config.boundmode))
  return new_vals


def setup_training(wandb_run):
  """Helper function that sets up training configs and logs to wandb."""
  if not wandb_run.config.get("use_tpu", False):
    # TF can hog GPU memory, so hide the GPU device from it
    # tf.config.experimental.set_visible_devices([], "GPU")

    # Without this, JAX is automatically using 90% GPU for pre-allocation
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
    # os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    # Disable logging of compiles
    jax.config.update("jax_log_compiles", False)

    # Log various JAX configs to wandb, and locally
    wandb_run.summary.update(
      {
        "jax_process_index": jax.process_index(),
        "jax.process_count": jax.process_count(),
      }
    )
  else:
    # config.FLAGS.jax_xla_backend = "tpu_driver"
    # config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
    # DEVICE_COUNT = len(jax.local_devices())
    print(jax.default_backend())
    print(jax.device_count(), jax.local_device_count())
    print("8 cores of TPU (Local devices in JAX):")
    print("\n".join(map(str, jax.local_devices())))


def main(config):
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
    new_vals = setup_config(run.config, config)
    update_config_dict(config, run, new_vals)
    print(config)
    assert 0

if __name__==main:
  main()
