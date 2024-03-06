"""Calculate the normalising constant using CMCD."""


FUNNEL_EPS_DICT = {
    8: {"init_eps": 0.1, "lr": 0.01},
    16: {"init_eps": 0.1, "lr": 0.01},
    32: {"init_eps": 0.1, "lr": 0.005},
    64: {"init_eps": 0.1, "lr": 0.001},
    128: {"init_eps": 0.01, "lr": 0.01},
    256: {"init_eps": 0.01, "lr": 0.005},
}


def load_model(model="funnel", config=None):
    d = config.funnel_d
    sig = config.funnel_sig
    clip_y = config.funnel_clipy

    def neg_energy(x):
        def unbatched(x):
            v = x[0]
            log_density_v = norm.logpdf(v, loc=0.0, scale=3.0)
            variance_other = np.exp(v)
            other_dim = d - 1
            cov_other = np.eye(other_dim) * variance_other
            mean_other = np.zeros(other_dim)
            log_density_other = multivariate_normal.logpdf(
                x[1:], mean=mean_other, cov=cov_other
            )
            return log_density_v + log_density_other

        output = np.squeeze(jax.vmap(unbatched)(x[None, :]))
        return output

    def sample_data(rng, n_samples):
        # sample from Nd funnel distribution

        y_rng, x_rng = jr.split(rng)

        y = (sig * jr.normal(y_rng, (n_samples, 1))).clip(-clip_y, clip_y)
        x = jr.normal(x_rng, (n_samples, d - 1)) * np.exp(-y / 2)
        return np.concatenate((y, x), axis=1)

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
    values = FUNNEL_EPS_DICT[wandb_config.num_outer_steps]
    new_vals = {"init_eps": values["init_eps"], "lr": values["lr"]}
    update_config_dict(config, run, new_vals)
    print(config)

  # Organize into a model setup thing in run_lib
  # If tractable distribution, we also return sample_from_target_fn
  log_prob_model, dim, sample_from_target_fn = load_model(
      config.model, config
  )


if __name__ == "__main__":
    app.run(main)
