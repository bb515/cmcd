"""Calculate the normalising constant using CMCD."""


def load_model(model="nice", config=None):
    artifact_name = f"{config.alpha}_{config.n_bits}_{config.im_size}"

    api = wandb.Api()

    artifact = api.artifact(f"shreyaspadhy/cais/{artifact_name}:latest")
    loaded_params = pickle.load(open(artifact.file(), "rb"))

    def forward_fn():
        flow = NICE(config.im_size**2, h_dim=config.hidden_dim)

        def _logpx(x):
            return flow.logpx(x)

        def _recons(x):
            return flow.reverse(flow.forward(x))

        def _sample(n):
            return flow.sample(n)

        return _logpx, (_logpx, _recons, _sample)

    forward = hk.multi_transform(forward_fn)

    logpx_fn, _, sample_fn = forward.apply

    logpx_fn_without_rng = lambda x: np.squeeze(
        logpx_fn(loaded_params, jax.random.PRNGKey(1), x[None, :])
    )

    sample_fn_clean = lambda rng, n: sample_fn(loaded_params, rng, n)

    return logpx_fn_without_rng, config.im_size**2, sample_fn_clean


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

  # jax_config.update("jax_traceback_filtering", "off")
  # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
  #
  #
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

  # Organize into a model setup thing in run_lib
  # If tractable distribution, we also return sample_from_target_fn
  log_prob_model, dim, sample_from_target_fn = load_model(
      config.model, config
  )

if __name__ == "__main__":
    app.run(main)
