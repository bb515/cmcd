"""Calculate the normalising constant using CMCD."""
import os
import pickle
from functools import partial
from diffusionjax.utils import flatten_nested_dict
from annealed_flow_transport.densities import LogDensity
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
import jax.scipy.linalg as slinalg
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import jax.random as random
import wandb


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", './configs/gmm.py', "Training configuration.",
  lock_config=True)
flags.DEFINE_string("workdir", './examples/', "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


class ChallengingTwoDimensionalMixture(LogDensity):
  """A challenging mixture of Gaussians in two dimensions.

  num_dim should be 2. config is unused in this case.
  """

  def _check_constructor_inputs(self, config,
                                sample_shape):
    del config
    # assert_trees_all_equal(sample_shape, (2,))

  def raw_log_density(self, x):
    """A raw log density that we will then symmetrize."""
    mean_a = jnp.array([3.0, 0.])
    mean_b = jnp.array([-2.5, 0.])
    mean_c = jnp.array([2.0, 3.0])
    means = jnp.stack((mean_a, mean_b, mean_c), axis=0)
    cov_a = jnp.array([[0.7, 0.], [0., 0.05]])
    cov_b = jnp.array([[0.7, 0.], [0., 0.05]])
    cov_c = jnp.array([[1.0, 0.95], [0.95, 1.0]])
    covs = jnp.stack((cov_a, cov_b, cov_c), axis=0)
    log_weights = jnp.log(jnp.array([1./3, 1./3., 1./3.]))
    l = jnp.linalg.cholesky(covs)
    y = slinalg.solve_triangular(l, x[None, :] - means, lower=True, trans=0)
    mahalanobis_term = -1/2 * jnp.einsum("...i,...i->...", y, y)
    n = means.shape[-1]
    normalizing_term = -n / 2 * jnp.log(2 * jnp.pi) - jnp.log(
        l.diagonal(axis1=-2, axis2=-1)).sum(axis=1)
    individual_log_pdfs = mahalanobis_term + normalizing_term
    mixture_weighted_pdfs = individual_log_pdfs + log_weights
    return logsumexp(mixture_weighted_pdfs)

  def make_2d_invariant(self, log_density, x):
    density_a = log_density(x)
    density_b = log_density(jnp.flip(x))
    return jnp.logaddexp(density_a, density_b) - jnp.log(2)

  def evaluate_log_density(self, x):
    density_func = lambda x: self.make_2d_invariant(self.raw_log_density, x)
    return density_func(x)
    # else: return jax.vmap(density_func)(x)

  def sample(self, rng_key, num_samples):
    mean_a = jnp.array([3.0, 0.0])
    mean_b = jnp.array([-2.5, 0.0])
    mean_c = jnp.array([2.0, 3.0])
    cov_a = jnp.array([[0.7, 0.0], [0.0, 0.05]])
    cov_b = jnp.array([[0.7, 0.0], [0.0, 0.05]])
    cov_c = jnp.array([[1.0, 0.95], [0.95, 1.0]])
    means = [mean_a, mean_b, mean_c]
    covs = [cov_a, cov_b, cov_c]
    log_weights = jnp.log(jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))
    num_components = len(means)
    samples = []
    k1, k2 = random.split(rng_key)
    # Sample from the GMM components based on the mixture weights
    for i, _ in enumerate(range(num_samples)):
      # Sample a component index based on the mixture weights
      component_idx = random.choice(
          k1 + i, num_components, p=jnp.exp(log_weights)
      )
      # Sample from the chosen component
      chosen_mean = means[component_idx]
      chosen_cov = covs[component_idx]
      sample = random.multivariate_normal(k2 + i, chosen_mean, chosen_cov)
      samples.append(sample)
    return jnp.stack(samples)


def load_model(config):
  sample_shape = (2,)
  gmm = ChallengingTwoDimensionalMixture(
    config,
    sample_shape=sample_shape,
  )
  return gmm.evaluate_log_density, gmm.sample, sample_shape


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  # num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

  # jax_config.update("jax_traceback_filtering", "off")
  # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

  log_prob_model, sample_from_target_fn, sample_shape = load_model(config)

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

    # Plot samples
    if config.model in ["nice", "funnel", "gmm"]:

      sample_from_target(config, sample_from_target_fn, samples, target_samples, n_samples)

      if config.model == "nice":

        make_grid(
          samples, config.im_size, n=64, wandb_prefix="images/final_sample"
        )
        if config.use_ema:
          make_grid(
            samples_ema,
            config.im_size,
            n=64,
            wandb_prefix="images/final_sample_ema",
          )

    if config.wandb.log_artifact:
      artifact_name = f"gmm_{config.solver.outer_solver}_{config.solver.num_outer_steps}"
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
