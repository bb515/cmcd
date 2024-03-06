"""Calculate the normalising constant using CMCD."""


def load_model(model="many_gmm", config=None):
  gmm = GMM(dim=2, n_mixes=config.n_mixes, loc_scaling=config.loc_scaling)
  return gmm.log_prob, 2, gmm.sample


class GMM:
  def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0):
    self.seed = seed
    self.n_mixes = n_mixes
    self.dim = dim
    key = jax.random.PRNGKey(seed)
    logits = np.ones(n_mixes)
    mean = (
        jax.random.uniform(shape=(n_mixes, dim), key=key, minval=-1.0, maxval=1.0)
        * loc_scaling
    )
    log_var = np.ones(shape=(n_mixes, dim)) * log_var_scaling

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
    log_prob = np.where(valid_log_prob, log_prob, -np.inf * np.ones_like(log_prob))
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
