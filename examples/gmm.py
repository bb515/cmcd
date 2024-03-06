"""Calculate the normalising constant using CMCD."""
from functools import partial
from diffusionjax.utils import flatten_nested_dict
from annealed_flow_transport.densities import LogDensity
from cmcd.utils import (
  calculate_W2_distances,
  )
from cmcd.solvers import MCD_SOLVERS
from cmcd.run_lib import (
  initialize,
  compute_bound,
  setup_training,
  get_solver,
  )
import cmcd.opt as opt
from ml_collections.config_flags import config_flags
from absl import app, flags
import jax
import jax.scipy.linalg as slinalg
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
    normalizing_term = -n / 2 * np.log(2 * np.pi) - jnp.log(
        l.diagonal(axis1=-2, axis2=-1)).sum(axis=1)
    individual_log_pdfs = mahalanobis_term + normalizing_term
    mixture_weighted_pdfs = individual_log_pdfs + log_weights
    return logsumexp(mixture_weighted_pdfs)

  def make_2d_invariant(self, log_density, x):
    density_a = log_density(x)
    density_b = log_density(np.flip(x))
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
          k1 + i, num_components, p=np.exp(log_weights)
      )
      # Sample from the chosen component
      chosen_mean = means[component_idx]
      chosen_cov = covs[component_idx]
      sample = random.multivariate_normal(k2 + i, chosen_mean, chosen_cov)
      samples.append(sample)
    return jnp.stack(samples)


def load_model(model="gmm", config=None):
  sample_shape = (2,)
  gmm = ChallengingTwoDimensionalMixture(
    config,
    sample_shape=sample_shape,
  )
  # log_density_fn = lambda x: np.squeeze(gmm.evaluate_log_density(x[None, :]))
  # x = np.array([0., 0.])
  # print(x.shape)
  # print(gmm.evaluate_log_density(np.array([0., 0.])))
  return gmm.evaluate_log_density, gmm.sample, sample_shape


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

  # jax_config.update("jax_traceback_filtering", "off")
  # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

  wandb_kwargs = {
    "project": config.wandb.project,
    "entity": config.wandb.entity,
    "config": flatten_nested_dict(config.to_dict()),
    "name": config.wandb.name if config.wandb.name else None,
    "mode": "online" if config.wandb.log else "disabled",
    "settings": wandb.Settings(code_dir=config.wandb.code_dir),
  }

  # TODO: optimally place in a :meth:train in run_lib but after I know is working
  with wandb.init(**wandb_kwargs) as run:
    setup_training(run)

    # Organize into a model setup thing in run_lib
    # If tractable distribution, we also return sample_from_target_fn
    log_prob_model, sample_from_target_fn, sample_shape = load_model(
        config.model, config)
    # (dim,) = sample_shape

    # Set up random seeds
    rng_key_gen = jax.random.PRNGKey(config.seed)
    train_rng_key_gen, eval_rng_key_gen = jax.random.split(rng_key_gen)

    # Train initial variational distribution to maximize the ELBO
    trainable = ("vd",)
    # TODO:
    params_flat, unflatten, params_fixed = initialize(
      config,
      shape,
      trainable=trainable,
      init_sigma=config.solver.init_sigma)

    shape = (config.training.batch_size,) + sample_shape

    compute_bound_fn = partial(
      compute_bound,
      config=config,
      shape=shape,
    )
    grad_and_loss = jax.jit(
      jax.grad(compute_bound_fn, 1, has_aux=True), static_argnums=(2, 3, 4))
    if not config.mfvi.pretrain:
      mfvi_iters = 1
      vdparams_init = unflatten(params_flat)[0]["vd"]
    else:
      losses, params_flat, _ = opt.run(
        config,
        config.mfvi.lr,
        config.mfvi.iters,
        params_flat,
        unflatten,
        params_fixed,
        log_prob_model,
        grad_and_loss,
        trainable,
        train_rng_key_gen,
        log_prefix="pretrain",
        use_ema=False,
      )
      vdparams_init = unflatten(params_flat)[0]["vd"]
      elbo_init = -jnp.mean(jnp.array(losses[-500:]))
      print("Done training initial parameters, got ELBO %.2f." % elbo_init)
      wandb.log({"elbo_init": np.array(elbo_init)})

    if config.solver.outer_solver == "UHA":  # TODO: this is UHA from src/ais_utils.py?
      trainable = ("eta", "mgridref_y")
      if config.train_eps:
        trainable = trainable + ("eps",)
      if config.train_vi:
        trainable = trainable + ("vd",)
      params_flat, unflatten, params_fixed = initialize(
        config,
        vdparams=vdparams_init,
        mdparams=mdparams,
        trainable=trainable)

      # TODO: sample shape depends on the example, so cannot be factored out of this example
      shape = (config.training.num_batch, config.training.image_size)

      grad_and_loss = jax.jit(
          jax.grad(compute_bound, 1, has_aux=True), static_argnums=(2, 3, 4), config=config, shape=shape,
      )
      loss = jax.jit(compute_bound, static_argnums=(2, 3, 4))

    elif config.solver.outer_solver in MCD_SOLVERS:
      trainable = ("eta", "gamma", "mgridref_y")
      if config.train_eps:
        trainable = trainable + ("eps",)
      if config.train_vi:
        trainable = trainable + ("vd",)

      print(f"Params being trained : {trainable}")
      params_flat, unflatten, params_fixed = initialize(
        config=config,
        vdparams=vdparams_init,
        trainable=trainable,
        solver=config.solver.outer_solver,
      )

      if "var" in config.solver:
        compute_bound_fn = partial(
          compute_bound_var,
          config=config,
          shape=shape,
        )
      else:
        compute_bound_fn = partial(
          compute_bound,
          config=config,
          shape=shape,
        )

      grad_and_loss = jax.jit(
        jax.grad(compute_bound_fn, 1, has_aux=True), static_argnums=(2, 3, 4),
      )
      loss_fn = jax.jit(compute_bound_fn, static_argnums=(2, 3, 4))

    else:
      raise NotImplementedError("Mode %s not implemented." % config.boundmode)

    # Average over 30 seeds, 500 samples each after training is done.
    n_samples = config.n_samples
    n_input_dist_seeds = config.n_input_dist_seeds

    if sample_from_target_fn is not None:
      target_samples = sample_from_target_fn(
        jax.random.PRNGKey(1), n_samples * n_input_dist_seeds
      )
    else:
      target_samples = None

    _, params_flat, ema_params = opt.run(
      config,
      config.lr,
      config.iters,
      params_flat,
      unflatten,
      params_fixed,
      log_prob_model,
      grad_and_loss,
      trainable,
      train_rng_key_gen,
      log_prefix="train",
      target_samples=target_samples,
      use_ema=config.use_ema,
    )

    eval_losses, samples = opt.sample(
      config,
      n_samples,
      n_input_dist_seeds,
      params_flat,
      unflatten,
      params_fixed,
      log_prob_model,
      loss_fn,
      eval_rng_key_gen,
      log_prefix="eval",
    )

    final_elbo, final_ln_Z = log_final_losses(eval_losses)

    print("Done training, got ELBO %.2f." % final_elbo)
    print("Done training, got ln Z %.2f." % final_ln_Z)

    if config.use_ema:
      eval_losses_ema, samples_ema = opt.sample(
        config,
        n_samples,
        n_input_dist_seeds,
        ema_params,
        unflatten,
        params_fixed,
        log_prob_model,
        loss_fn,
        eval_rng_key_gen,
        log_prefix="eval",
      )

      final_elbo_ema, final_ln_Z_ema = log_final_losses(
        eval_losses_ema, log_prefix="_ema"
      )

      print("With EMA, got ELBO %.2f." % final_elbo_ema)
      print("With EMA, got ln Z %.2f." % final_ln_Z_ema)

    # TODO: not sure if it is best to have different example for each file or have them use same
    # TODO: extract useful setup functions if were to create an example from scratch.
    # Plot samples
    if config.model in ["nice", "funnel", "gmm"]:
      other_target_samples = sample_from_target_fn(
        jax.random.PRNGKey(2), samples.shape[0]
      )

      calculate_W2_distances(
        samples,
        target_samples,
        other_target_samples,
        n_samples,
        config.n_input_dist_seeds,
        n_samples,
      )

      if config.use_ema:
        calculate_W2_distances(
          samples_ema,
          target_samples,
          other_target_samples,
          n_samples,
          config.n_input_dist_seeds,
          n_samples,
          log_prefix="_ema",
        )

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

    params_train, params_notrain = unflatten(params_flat)
    params = {**params_train, **params_notrain}

    if config.wandb.log_artifact:
      artifact_name = f"{config.model}_{config.boundmode}_{config.nbridges}"
      artifact = wandb.Artifact(
        artifact_name,
        type="final params",
      )
      # Save model
      with artifact.new_file("params.pkl", "wb") as f:
        pickle.dump(params, f)

      wandb.log_artifact(artifact)

if __name__ == "__main__":
    app.run(main)
