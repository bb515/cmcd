import jax
import jax.numpy as jnp
import jax.random as random
from jax.lax import scan
from jax import jit, grad, value_and_grad
import numpyro.distributions as npdist
from functools import partial
from diffusionjax.utils import get_times


def W2_distance(x, y, reg=0.01):
    N = x.shape[0]
    x, y = np.array(x), np.array(y)
    a, b = np.ones(N) / N, np.ones(N) / N

    M = ot.dist(x, y)
    M /= M.max()

    T_reg = ot.sinkhorn2(a, b, M, reg, log=False, numItermax=10000, stopThr=1e-16)
    return T_reg


def log_final_losses(eval_losses, log_prefix=""):
    """
    eval_losses is of shape (n_input_dist_seeds, n_samples)
    """
    # (n_input_dist_seeds, n_samples)
    eval_losses = jnp.array(eval_losses)
    n_samples = eval_losses.shape[1]
    # Calculate mean and std of ELBOs over 30 seeds
    final_elbos = -jnp.mean(eval_losses, axis=1)
    final_elbo = jnp.mean(final_elbos)
    final_elbo_std = jnp.std(final_elbos)

    # Calculate mean and std of log Zs over 30 seeds
    ln_numsamp = jnp.log(n_samples)

    final_ln_Zs = jscipy.special.logsumexp(-jnp.array(eval_losses), axis=1) - ln_numsamp

    final_ln_Z = jnp.mean(final_ln_Zs)
    final_ln_Z_std = jnp.std(final_ln_Zs)

    wandb.log(
        {
            f"elbo_final{log_prefix}": np.array(final_elbo),
            f"final_ln_Z{log_prefix}": np.array(final_ln_Z),
            f"elbo_final_std{log_prefix}": np.array(final_elbo_std),
            f"final_ln_Z_std{log_prefix}": np.array(final_ln_Z_std),
        }
    )

    return final_elbo, final_ln_Z


def calculate_W2_distances(
  samples,
  target_samples,
  other_target_samples,
  n_samples,
  n_input_dist_seeds,
  n_sinkhorn,
  log_prefix=""):
  w2_dists, self_w2_dists = [], []
  for i in range(n_input_dist_seeds):
    samples_i = samples[i * n_samples : (i + 1) * n_samples, ...]
    target_samples_i = target_samples[i * n_samples : (i + 1) * n_samples, ...]
    other_target_samples_i = other_target_samples[
      i * n_samples : (i + 1) * n_samples, ...]
    assert n_sinkhorn <= n_samples
    samples_i = samples_i[:n_sinkhorn, ...]
    target_samples_i = target_samples_i[:n_sinkhorn, ...]
    other_target_samples_i = other_target_samples_i[:n_sinkhorn, ...]

    w2_dists.append(W2_distance(samples_i, target_samples_i))
    self_w2_dists.append(W2_distance(target_samples_i, other_target_samples_i))

  wandb.log(
    {
      f"w2_dist{log_prefix}": np.mean(np.array(w2_dists)),
      f"w2_dist_std{log_prefix}": np.std(np.array(w2_dists)),
      f"self_w2_dist{log_prefix}": np.mean(np.array(self_w2_dists)),
      f"self_w2_dist_std{log_prefix}": np.std(np.array(self_w2_dists))})


def initialize_mcbd(
    config, dim, vdparams=None, eps=0.01, gamma=10., eta=.5, ngridb=32, mgridref_y=None, trainable=["eps"], use_score_nn=True,
    emb_dim=48, nlayers=3, seed=1, mode="MCD_U_lp-e"):
  """
  Modes allowed:
    - MCD_ULA: This is ULA. Method from Thin et al.
    - MCD_ULA_sn: This is MCD. Method from Doucet et al.
    - MCD_U_a-lp: UHA but with approximate sampling of momentum (no score network).
    - MCD_U_a-lp-sn: Approximate sampling of momentum, followed by leapfrog, using score network(x, rho) for backward sampling.
    - MCD_CAIS_sn: CAIS with trainable SN.
    - MCD_CAIS_UHA_sn: CAIS underdampened with trainable SN.
  """
  params_train = {}
  params_notrain = {}
  for param in ["vd", "eps", "eta", "md"]:
    if param in trainable:
      init_param = config.repr(param)
      if init_param is None:
        if param is "vd": init_param = vd.initialize(dim, init_sigma=init_sigma)
        if param is "md": init_param = md.initialize(dim)
      params_train[param] = init_param
    else:
      params_notrain[param] = init_param

  rng = jax.random.PRNGKey(config.seed)
  if mode in [
    "ULA_sn",
    "U_elpsna",
    "U_alpsna",
    "CMCD_sn",  # Overdamped?
    "CMCD_var_sn",
  ]:
    init_sn, apply_sn = initialize_mcd_network(dim, emb_dim, num_steps, num_layers=num_layers)
    params_train["sn"] = init_sn(rng, None)[1]
  elif mode in [
      ".",
    ]:
    init_sn, apply_sn = initialize_mcd_network(dim, emb_dim, num_steps, xd_dim=xd_dim, num_layers=num_layers)
    params_train["sn"] = init_sn(rng, None)[1]
  else:
    apply_sn = None
    print("No score network needed by the method.")

  # Everything related to betas
  target_x, gridref_x, mgridref_y, ts = get_betas(config.solver.num_steps)
  params_notrain["gridref_x"] = gridref_x
  # BB: Does it not make sense to start this at 1. since density must be evaluated at 1. where the prior sample is initiated.
  # at the distribution that one starts in? i.e., shouldn't it instead be: # params_notrain["target_x"] = jnp.linspace(0, 1, num_steps + 1)[0:-1]
  params_notrain["target_x"] = target_x

  if "mgridref_y" in trainable:
    params_train["mgridref_y"] = mgridref_y
  else:
    params_notrain["mgridref_y"] = mgridref_y

  # Other fixed parameters
  params_fixed = (dim, num_step)
  params_flat, unflatten = ravel_pytree((params_train, params_notrain))
  return params_flat, unflatten, params_fixed


def initialize(
  config,
  dim,
  vdparams=None,
):
  dim = config.image_size
  params_train = {}  # all trainable parameters
  params_notrain = {}  # all non-trainable parameters
  for param in ["vd", "eps", "eta", "md"]:
    if param in trainable:
      init_param = config.repr(param)
      if init_param is None:
        if param is "vd": init_param = vd.initialize(dim, init_sigma=init_sigma)
        if param is "md": init_param = md.initialize(dim)
      params_train[param] = init_param
    else:
      params_notrain[param] = init_param

  # Everything related to betas
  target_x, gridref_x, mgridref_y, ts = get_betas(config.solver.num_steps)
  params_notrain["gridref_x"] = gridref_x
  # BB: Does it not make sense to start this at 1. since density must be evaluated at 1. where the prior sample is initiated.
  # at the distribution that one starts in? i.e., shouldn't it instead be: # params_notrain["target_x"] = jnp.linspace(0, 1, num_steps + 1)[0:-1]
  params_notrain["target_x"] = target_x

  if "mgridref_y" in trainable:
    params_train["mgridref_y"] = mgridref_y
  else:
    params_notrain["mgridref_y"] = mgridref_y

  # Other fixed parameters
  params_fixed = (dim, num_step)
  params_flat, unflatten = ravel_pytree((params_train, params_notrain))
  return params_flat, unflatten, params_fixed


def initialize_bm():
  # Other fixed parameters
  params_fixed = (dim, nbridges, lfsteps)
  params_flat, unflatten = ravel_pytree((params_train, params_notrain))
  if nbridges >= 1:
      gridref_y = np.cumsum(params["mgridref_y"]) / np.sum(params["mgridref_y"])
      print(gridref_y)
      gridref_y = np.concatenate([np.array([0.0]), gridref_y])
      betas = np.interp(params["target_x"], params["gridref_x"], gridref_y)

  return None


def compute_ratio_mcbm(seed, params_flat, unflatten, params_fixed, log_prob, Solver):
  params_train, params_notrain = unflatten(params_flat)
  params_notrain = jax.lax.stop_gradient(params_notrain)
  params = {**params_train, **params_notrain}  # All parameters in a single place
  dim, num_steps, _, _ = params_fixed

  if num_steps >= 1:
    target_x, gridref_x, mgridref_y, ts = get_betas(config.solver.num_steps)

  # NOTE Shreyas said this was computed outside the solver, here it is. But it can be
  # computed inside the sampler
  # But why is this variational distribution, is that right?
  # rng = jax.random.PRNGKey(seed)
  # rng, step_rng = jax.random.split(rng, 2)
  # w = -vd.log_prob(params["vd"], x)

  # Evolve sampler and update weight
  sampler = get_sampler(shape, outer_solver, denoise=config.sampling.denoise,
                        stack_samples=False, inverse_scaler=inverse_scaler)
  rng, sample_rng = jax.random.split(rng, 2)

  q_samples, aux, num_function_evaluations = sampler(sample_rng)
  w, delta_H = aux
  delta_H = jnp.max(jnp.abs(delta_H))
  # delta_H = jnp.mean(jnp.abs(delta_H))
  return -1. * w, (x,)


def compute_bound_bm():
  pass


def compute_ratio_bm(seed, params_flat, unflatten, params_fixed, log_prob, Solver):
  # TODO: this is like a sampler, but keeps parameters in one place, I think that's a good idea.
  # but it would be weird to train the beta schedule I think - but maybe that's the point of the work
  #
  #
  # If params not supplied here:
  # params_flat, params_fixed = initialize_bm()

  params_train, params_notrain = unflatten(params_flat)
  params_notrain = jax.lax.stop_gradient(params_notrain)
  params = {**params_train, **params_notrain}  # Gets all parameters in single place
  dim, num_steps, lfsteps = params_fixed

  # Need to reinitiate solver for every score
  target_x, gridref_x, mgridref_y, ts = get_betas(config.num_steps)
  # ts = get_times(num_steps)
  # beta = get_linear_beta_schedule(ts)

  auxilliary_process_score = get_score(sde, model, params, score_scaling)
  outer_solver = Solver(params, base_process_score, auxilliary_process_score, beta, ts)

  rng_key_gen = jax.random.PRNGKey(config.seed)
  rng, sample_rng = jax.random.split(rng, 2)
  sampler = get_sampler(shape, outer_solver, denoise=False)
  x, x_mean, aux = sampler(sample_rng)

  # Update weight with final model evaluation
  w = w + log_prob(z)
  delta_H = np.max(np.abs(delta_H))
  # delta_H = np.mean(np.abs(delta_H))
  return -1.0 * w, (z, delta_H)


def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob):
    ratios, (z, _) = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob)
    return ratios.mean(), (ratios, z)


def get_cosine_beta_schedule(num_steps, dt, t0, beta_min=.1, beta_max=20., offset=0.08):
  phases, dt = get_times(num_steps, dt, t0)
  beta, _ = get_cosine_beta_function(offset)
  decay = vmap(beta)(phases)
  beta, _ = get_linear_beta_function(beta_min, beta_max)
  continuous_betas = vmap(beta)(decay)
  discrete_betas = continuous_to_discrete(continuous_betas, dt)  # Adaptive timestepping
  return discrete_betas


def get_linear_beta_schedule(num_steps, dt, t0, beta_min=1., beta_max=20.):
  ts, dt = get_times(num_steps, dt, t0)
  beta, _ = get_linear_beta_function(beta_min, beta_max)
  continuous_betas = vmap(beta)(ts)
  discrete_betas = continuous_to_discrete(continuous_betas, dt)  # Adaptive timestepping
  return discrete_betas

def log_prob_kernel(x, mean, scale):
  """For evaluating Markov transition kernels. Not the same as sampling,
  and so no computation should be wasted, and good to use a standardised library."""
  dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
  return dist.log_prob(x)

def sample_rep(rng, params):
  mean, logdiag = params["mean"], params["logdiag"]
  d_x = mean.shape[0]
  z = jax.random.normal(rng, shape=(dim,))
  return mean + z * jnp.exp(log_diag)

def build(params):
  mean, logdiag = params["mean"], params["logdiag"]
  return npdist.Independent(npdist.Normal(loc=mean, scale=jnp.exp(logdiag)), 1)

def entropy(params):
  dist = build(params)
  return dist.entropy()


def reparameterize(params, eps):
  mean, logdiag = decode_params(params)
  return to_scale(logdiag) * eps + mean


def sample_rep(rng_key, params):
  mean, _ = decode_params(params)
  dim = mean.shape[0]
  eps = sample_eps(rng_key, dim)
  eps = jax.random.normal(rng_key, shape=(dim,))
  return reparameterize(params, eps)


def initialize(dim, init_sigma=1.0):
  mean = jnp.zeros(dim)
  logdiag = jnp.ones(dim) * jnp.log(init_sigma)
  return {"mean": mean, "logdiag": logdiag}


def log_prob(vdparams, z):
  dist = build(params)
  return dist.log_prob(z)


def log_prob_frozen(z, params):
  dist = build(jax.lax.stop_gradient(params))
  return dist.log_prob(z)


def get_betas(num_steps=1000):
  # Everything related to betas
  ngridb = num_steps
  import matplotlib.pyplot as plt
  mgridref_y = jnp.ones(ngridb + 1) * 1.0
  gridref_x = jnp.linspace(0, 1, num_steps + 2)
  # BB: Does it not make sense to start this at 1. since density must be evaluated at 1. where the prior sample is initiated.
  # at the distribution that one starts in? i.e., shouldn't it instead be: # params_notrain["target_x"] = jnp.linspace(0, 1, num_steps + 1)[0:-1]
  target_x = jnp.linspace(0, 1, num_steps + 2)[1:-1]
  # return mgridref_y, gridref_x, target_x
  # BB: I don't understand the purpose of gridref_y. Seems to be redundant since
  # y = f(x) is a linear relationship.
  # I guess that if a cosine schedule is used, then this would define that transform?
  gridref_y = jnp.cumsum(mgridref_y) / jnp.sum(mgridref_y)
  gridref_y = jnp.concatenate([jnp.array([0.0]), gridref_y])

  # Interpolates the function that takes gridref_x -> gridref_y, evaluated at target_x
  # the reason for this is because they will treat t=1.0 and t=0.0 differently
  ts = jnp.interp(target_x, gridref_x, gridref_y)
  dt = ts[1] - ts[0]

  # BB: Why not just have
  ts_alt = gridref_y[1:-1]
  dt_alt = ts_alt[1] - ts_alt[0]
  ts_bb, dt_bb = get_times(num_steps)
  print("dt: ", dt_bb, dt, dt_alt)
  print("len: ", ts.shape, ts_alt.shape, ts_bb.shape)
  print("maxmin", jnp.max(ts), jnp.min(ts))
  print(jnp.max(ts_bb), jnp.min(ts_bb))
  import matplotlib.pyplot as plt
  plt.plot(ts, ts_alt)
  plt.savefig("testnative.png")
  plt.close()
  plt.plot(ts, ts_bb.flatten() / ts)
  plt.savefig("testbrel.png")
  plt.close()
  plt.plot(ts, ts_bb.flatten() - ts)
  plt.savefig("testbabs.png")
  plt.close()
  assert jnp.allclose(ts, ts_alt)
  assert jnp.allclose(dt, dt_alt)
  assert jnp.allclose(ts, ts_bb)
  assert jnp.allclose(dt, dt_bb)
  return target_x, gridref_x, mgridref_y, ts


def get_annealed_langevin(log_prob_model):
  def potential(x, t):
      return -1. * (
          t * log_prob_model(x) + (1. - t) * log_prob(params["vd"], z)
      )


log_2pi = jnp.log(2 * jnp.pi)


def log_prob_kernel(x, mean, scale):
  """Unit test for numpyro."""
  n = jnp.size(x)
  # deal with zero variances by adding a constant term if zero
  var = scale**2
  centered = x - mean
  return - 0.5 * (n * log_2pi + jnp.sum(jnp.log(var)) + centered.T @ centered / var)


def shared_update(rng, x, t, aux, solver, probability_flow=None):
  """A wrapper that configures and returns the update function of the solvers.

  :probablity_flow: Placeholder for probability flow ODE (TODO).
  """
  return solver.update(rng, x, t, aux)


def get_underdamped_sampler(shape, outer_solver, inner_solver=None, denoise=True, stack_samples=False, inverse_scaler=None):
  """Get a sampler from (possibly interleaved) numerical solver(s).

  Args:
    shape: Shape of array, x. (num_samples,) + obj_shape, where x_shape is the shape
      of the object being sampled from, for example, an image may have
      obj_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
    outer_solver: A valid numerical solver class that will act on an outer loop.
    inner_solver: '' that will act on an inner loop.
    denoise: Bool, that if `True` applies one-step denoising to final samples.
    stack_samples: Bool, that if `True` return all of the sample path or
      just returns the last sample.
    inverse_scaler: The inverse data normalizer function.
  Returns:
    A sampler.
  """
  if inverse_scaler is None: inverse_scaler = lambda x: x

  def sampler(rng, x_0=None, aux_0=None):
    """
    Args:
      rng: A JAX random state.
      x_0: Initial condition. If `None`, then samples an initial condition from the
          sde's initial condition prior. Note that this initial condition represents
          `x_T sim Normal(O, I)` in reverse-time diffusion.
    Returns:
        Samples and the number of score function (model) evaluations and auxiliary variables,
        like weight from transition kernels, normalizing constant,
        and forwards/backwards kernel auxiliary.
    """
    outer_update = partial(shared_update,
                           solver=outer_solver)
    outer_ts = outer_solver.ts

    if inner_solver:
        inner_update = partial(shared_update,
                               solver=inner_solver)
        inner_ts = inner_solver.ts
        num_function_evaluations = jnp.size(outer_ts) * (jnp.size(inner_ts) + 1)

        def inner_step(carry, t):
          rng, x, x_d, vec_t, aux = carry
          rng, step_rng = random.split(rng)
          x, x_d, aux = inner_update(step_rng, x, vec_t, aux)
          return (rng, x, x_mean, vec_t, aux), ()

        def outer_step(carry, t):
          rng, x, x_mean, aux = carry
          vec_t = jnp.full(shape[0], t)
          rng, step_rng = random.split(rng)
          x, x_mean, aux = outer_update(step_rng, x, vec_t)
          (rng, x, x_mean, vec_t, aux), _ = scan(inner_step, (step_rng, x, x_mean, vec_t, aux), inner_ts)
          if not stack_samples:
            return (rng, x, x_mean, aux), ()
          else:
            if denoise:
              return (rng, x, x_mean, aux), x_mean
            else:
              return (rng, x, x_mean, aux), x
    else:
      num_function_evaluations = jnp.size(outer_ts)
      def outer_step(carry, t):
        rng, x, x_mean, aux = carry
        vec_t = jnp.full((shape[0],), t)
        rng, step_rng = random.split(rng)
        x, x_mean, aux = outer_update(step_rng, x, vec_t, aux)
        if not stack_samples:
          return (rng, x, x_mean, aux), ()
        else:
          return ((rng, x, x_mean, aux), x_mean) if denoise else ((rng, x, x_mean, aux), x)

    rng, step_rng = random.split(rng)
    # x_0 is initiated to zero in these underdamped solvers?
    if x_0 is None:
      if inner_solver:
        x = inner_solver.prior(step_rng, shape)
      else:
        x = outer_solver.prior(step_rng, shape)
    else:
      assert(x_0.shape==shape)
      x = x_0
    rng, step_rng = random.split(rng)
    if xd_0 is None:
      if inner_solver:
        xd = inner_solver.prior(step_rng, shape)
      else:
        xd = outer_solver.prior(step_rng, shape)
    else:
      assert(xd_0.shape==shape)
      xd = xd_0
    if aux_0 is None:
      if inner_solver:
        aux = inner_solver.init_aux(xd)
      else:
        aux = outer_solver.init_aux(xd)
    else:
        aux = aux_0

    dist = npdist.Independent(npdist.Normal(loc=jnp.zeros(xd.shape), scale=1.), 1)
    w = w + dist.log_prob(xd)

    if not stack_samples:
      (_, x, x_mean, aux), _ = scan(outer_step, (rng, x, xd, aux), outer_ts, reverse=True)
      aux = outer_solver.fin_aux(aux, shape)
      return inverse_scaler(x_mean if denoise else x), aux, num_function_evaluations
    else:
      (_, _, _, _), xs = scan(outer_step, (rng, x, xd, aux), outer_ts, reverse=True)
      aux = outer_solver.fin_aux(aux, shape)
      return inverse_scaler(xs), aux, num_function_evaluations

  # return jax.pmap(sampler, in_axes=(0), axis_name='batch')
  return sampler


def get_overdamped_sampler(shape, outer_solver, inner_solver=None, denoise=True, stack_samples=False, inverse_scaler=None):
  """Get a sampler from (possibly interleaved) numerical solver(s).

  Args:
    shape: Shape of array, x. (num_samples,) + obj_shape, where x_shape is the shape
      of the object being sampled from, for example, an image may have
      obj_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
    outer_solver: A valid numerical solver class that will act on an outer loop.
    inner_solver: '' that will act on an inner loop.
    denoise: Bool, that if `True` applies one-step denoising to final samples.
    stack_samples: Bool, that if `True` return all of the sample path or
      just returns the last sample.
    inverse_scaler: The inverse data normalizer function.
  Returns:
    A sampler.
  """
  if inverse_scaler is None: inverse_scaler = lambda x: x

  def sampler(rng, x_0=None, aux_0=None):
    """
    Args:
      rng: A JAX random state.
      x_0: Initial condition. If `None`, then samples an initial condition from the
          sde's initial condition prior. Note that this initial condition represents
          `x_T sim Normal(O, I)` in reverse-time diffusion.
    Returns:
        Samples and the number of score function (model) evaluations and auxiliary variables,
        like weight from transition kernels, normalizing constant,
        and forwards/backwards kernel auxiliary.
    """
    outer_update = partial(shared_update,
                           solver=outer_solver)
    outer_ts = outer_solver.ts

    if inner_solver:
        inner_update = partial(shared_update,
                               solver=inner_solver)
        inner_ts = inner_solver.ts
        num_function_evaluations = jnp.size(outer_ts) * (jnp.size(inner_ts) + 1)

        def inner_step(carry, t):
          rng, x, x_mean, vec_t, aux = carry
          rng, step_rng = random.split(rng)
          x, x_mean, aux = inner_update(step_rng, x, vec_t, aux)
          return (rng, x, x_mean, vec_t, aux), ()

        def outer_step(carry, t):
          rng, x, x_mean, aux = carry
          vec_t = jnp.full(shape[0], t)
          rng, step_rng = random.split(rng)
          x, x_mean, aux = outer_update(step_rng, x, vec_t)
          (rng, x, x_mean, vec_t, aux), _ = scan(inner_step, (step_rng, x, x_mean, vec_t), inner_ts)
          if not stack_samples:
            return (rng, x, x_mean, aux), ()
          else:
            if denoise:
              return (rng, x, x_mean, aux), x_mean
            else:
              return (rng, x, x_mean, aux), x
    else:
      num_function_evaluations = jnp.size(outer_ts)
      def outer_step(carry, t):
        rng, x, x_mean, aux = carry
        vec_t = jnp.full((shape[0],), t)
        rng, step_rng = random.split(rng)
        x, x_mean, aux = outer_update(step_rng, x, vec_t, aux)
        if not stack_samples:
          return (rng, x, x_mean, aux), ()
        else:
          return ((rng, x, x_mean, aux), x_mean) if denoise else ((rng, x, x_mean, aux), x)

    rng, step_rng = random.split(rng)

    if x_0 is None:
      if inner_solver:
        x = inner_solver.prior(step_rng, shape)
      else:
        x = outer_solver.prior(step_rng, shape)
    else:
      assert(x_0.shape==shape)
      x = x_0

    if aux_0 is None:
      if inner_solver:
        aux = inner_solver.init_aux(step_rng, shape)
      else:
        aux = outer_solver.init_aux(step_rng, shape)
    else:
        aux = aux_0

    if not stack_samples:
      (_, x, x_mean, aux), _ = scan(outer_step, (rng, x, x, aux), outer_ts, reverse=True)
      aux = outer_solver.fin_aux(aux)
      return inverse_scaler(x_mean if denoise else x), aux, num_function_evaluations
    else:
      (_, _, _, _), xs = scan(outer_step, (rng, x, x, aux), outer_ts, reverse=True)
      aux = outer_solver.fin_aux(aux)
      return inverse_scaler(xs), aux, num_function_evaluations

  # return jax.pmap(sampler, in_axes=(0), axis_name='batch')
  return sampler


# TODO move this to unit test
get_betas(num_steps=1000)

# from jax import vmap
# num_steps = 1000
# t1 = 1.0
# t0 = 0.001
# dt = 0.001
# offset = 0.8
# ii = jnp.linspace(0, 999, num_steps, dtype=int)
# print(ii)
# print(dt)

# SSlinear = vmap(lambda i: SSlinear_beta_schedule(num_steps, t1, i, final_eps=t0))(ii)
# SScosine = vmap(lambda i: SScosine_beta_schedule(num_steps, t1, i, s=offset))(ii)

# linear, _dt, _t0, _t1 = get_linear_beta_schedule(num_steps, dt, t0)
# cosine, _dt, _t0, _t1 = get_cosine_beta_schedule(num_steps, dt, t0, offset)
# linear = linear.flatten()
# cosine = cosine.flatten()
# # print(linear)
# # print(cosine)
# # print(SSlinear)
# # print(SScosine)

# import matplotlib.pyplot as plt

# plt.plot(ii, SSlinear)
# plt.savefig("SSlinear.png")
# plt.close()

# plt.plot(ii, SScosine)
# plt.savefig("SScosine.png")
# plt.close()

# plt.plot(ii, linear)
# plt.savefig("linear.png")
# plt.close()

# plt.plot(ii, cosine)
# plt.savefig("cosine.png")
# plt.close()

# betas = get_betas(num_steps=1000)
# print(betas)
