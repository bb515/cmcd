"""Calculate the normalising constant using CMCD."""
import inference_gym.using_jax as gym


models_gym = ["lorenz", "brownian", "banana"]
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
    return log_prob_model, dim


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

  # jax_config.update("jax_traceback_filtering", "off")
  # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

  ml_collections.config_flags.DEFINE_config_file(
      "config",
      "configs/base.py",
      "Training configuration.",
      lock_config=False,
  )
  FLAGS = flags.FLAGS

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

  # Organize into a model setup thing in run_lib
  # If tractable distribution, we also return sample_from_target_fn
  log_prob_model, dim = load_model(config.model, config)
  sample_from_target_fn = None


if __name__ == "__main__":
    app.run(main)
