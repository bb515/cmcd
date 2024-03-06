"""Calculate the normalising constant using CMCD."""
from annealed_flow_transport.densities import LogDensity


class LogGaussianCoxPines(LogDensity):
    """Log Gaussian Cox process posterior in 2D for pine saplings data.

    This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

    config.file_path should point to a csv file of num_points columns
    and 2 rows containg the Finnish pines data.

    config.use_whitened is a boolean specifying whether or not to use a
    reparameterization in terms of the Cholesky decomposition of the prior.
    See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
    The experiments in the paper have this set to False.

    num_dim should be the square of the lattice sites per dimension.
    So for a 40 x 40 grid num_dim should be 1600.
    """

    def __init__(self, config: ConfigDict, num_dim: int = 1600):
        super().__init__(config, num_dim)

        # Discretization is as in Controlled Sequential Monte Carlo
        # by Heng et al 2017 https://arxiv.org/abs/1708.08396
        self._num_latents = num_dim
        self._num_grid_per_dim = int(np.sqrt(num_dim))

        bin_counts = np.array(
            cp_utils.get_bin_counts(
                self.get_pines_points(config.file_path), self._num_grid_per_dim
            )
        )

        self._flat_bin_counts = np.reshape(bin_counts, (self._num_latents))

        # This normalizes by the number of elements in the grid
        self._poisson_a = 1.0 / self._num_latents
        # Parameters for LGCP are as estimated in Moller et al, 1998
        # "Log Gaussian Cox processes" and are also used in Heng et al.

        self._signal_variance = 1.91
        self._beta = 1.0 / 33

        self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

        def short_kernel_func(x, y):
            return cp_utils.kernel_func(
                x, y, self._signal_variance, self._num_grid_per_dim, self._beta
            )

        self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
        self._cholesky_gram = np.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = (
            -0.5 * self._num_latents * np.log(2.0 * np.pi)
        )

        half_log_det_gram = np.sum(np.log(np.abs(np.diag(self._cholesky_gram))))
        self._unwhitened_gaussian_log_normalizer = (
            -0.5 * self._num_latents * np.log(2.0 * np.pi) - half_log_det_gram
        )
        # The mean function is a constant with value mu_zero.
        self._mu_zero = np.log(126.0) - 0.5 * self._signal_variance

        if self._config.use_whitened:
            self._posterior_log_density = self.whitened_posterior_log_density
        else:
            self._posterior_log_density = self.unwhitened_posterior_log_density

    def _check_constructor_inputs(self, config: ConfigDict, num_dim: int):
        expected_members_types = [("use_whitened", bool)]
        self._check_members_types(config, expected_members_types)
        num_grid_per_dim = int(np.sqrt(num_dim))
        if num_grid_per_dim * num_grid_per_dim != num_dim:
            msg = (
                "num_dim needs to be a square number for LogGaussianCoxPines "
                "density."
            )
            raise ValueError(msg)

        if not config.file_path:
            msg = "Please specify a path in config for the Finnish pines data csv."
            raise ValueError(msg)

    def get_pines_points(self, file_path):
        """Get the pines data points."""
        with open(file_path, mode="rt") as input_file:
            # with open(file_path, "rt") as input_file:
            b = onp.genfromtxt(input_file, delimiter=",")
        return b

    def whitened_posterior_log_density(self, white: Array) -> Array:
        quadratic_term = -0.5 * np.sum(white**2)
        prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
        latent_function = cp_utils.get_latents_from_white(
            white, self._mu_zero, self._cholesky_gram
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    def unwhitened_posterior_log_density(self, latents: Array) -> Array:
        white = cp_utils.get_white_from_latents(
            latents, self._mu_zero, self._cholesky_gram
        )
        prior_log_density = (
            -0.5 * np.sum(white * white) + self._unwhitened_gaussian_log_normalizer
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    def evaluate_log_density(self, x: Array) -> Array:
        # import pdb; pdb.set_trace()
        if len(x.shape) == 1:
            return self._posterior_log_density(x)
        else:
            return jax.vmap(self._posterior_log_density)(x)


def load_model(model="lgcp", config=None):
    lgcp = LogGaussianCoxPines(config, num_dim=1600)
    return lgcp.evaluate_log_density, lgcp._num_latents


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
