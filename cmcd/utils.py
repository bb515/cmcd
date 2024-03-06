import jax
import jax.numpy as jnp
import jax.random as random
from jax.lax import scan
from jax import jit, grad, value_and_grad
import numpyro.distributions as npdist
from functools import partial
from cmcd.nn import initialize_mcd_network


# TODO: replace with plot.py code
def make_grid(x, im_size: int, n: int = 16, wandb_prefix: str = ""):
    """
    Plot a grid of images, and optionally log to wandb.

    x: (N, im_size, im_size) array of images
    im_size: size of images
    n: number of images to plot
    wandb_prefix: prefix to use for wandb logging
    """
    x = np.array(x[:n].reshape(-1, im_size, im_size))

    n_rows = int(np.sqrt(n))
    fig, ax = plt.subplots(n_rows, n_rows, figsize=(8, 8))

    # Plot each image
    for i in range(n_rows):
        for j in range(n_rows):
            ax[i, j].imshow(x[i * n_rows + j], cmap="gray")
            ax[i, j].axis("off")

    # Log into wandb
    wandb.log({f"{wandb_prefix}": fig})
    plt.close()


def initialize_dist(dim, init_sigma=1.0):
    mean = jnp.zeros(dim)
    logdiag = jnp.ones(dim) * jnp.log(init_sigma)
    return {"mean": mean, "logdiag": logdiag}


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
  mean, log_diag = params["mean"], params["logdiag"]
  z = jax.random.normal(rng, shape=mean.shape)
  return mean + z * jnp.exp(log_diag)


def build(params):
  mean, logdiag = params["mean"], params["logdiag"]
  return npdist.Independent(npdist.Normal(loc=mean, scale=jnp.exp(logdiag)), 1)


def entropy(params):
  dist = build(params)
  return dist.entropy()


def log_prob(vdparams, z):
  dist = build(params)
  return dist.log_prob(z)


# TODO: needed?
def log_prob_frozen(z, params):
  dist = build(jax.lax.stop_gradient(params))
  return dist.log_prob(z)


def get_betas(num_steps):
  # Everything related to betas
  mgridref_y = jnp.ones(num_steps + 1) * 1.0
  gridref_x = jnp.linspace(0, 1, num_steps + 2)

  # BB: Does it not make sense to start this at 1. since density must be evaluated at 1. where the prior sample is initiated.
  # at the distribution that one starts in? i.e., shouldn't it instead be: # params_notrain["target_x"] = jnp.linspace(0, 1, num_steps + 1)[0:-1]
  target_x = jnp.linspace(0, 1, num_steps + 2)[1:-1]

  # BB: I don't understand the purpose of gridref_y. Seems to be redundant since
  # y = f(x) is a linear relationship.
  # I guess that if a cosine schedule is used, then this would define that transform?
  gridref_y = jnp.cumsum(mgridref_y) / jnp.sum(mgridref_y)
  gridref_y = jnp.concatenate([jnp.array([0.0]), gridref_y])

  # Interpolates the function that takes gridref_x -> gridref_y, evaluated at target_x
  # the reason for this is because they will treat t=1.0 and t=0.0 differently
  ts = jnp.interp(target_x, gridref_x, gridref_y)
  dt = ts[1] - ts[0]
  return target_x, gridref_x, mgridref_y, ts


def get_annealed_langevin(log_prob_model):
  def potential(x, t):
      return -1. * (t * log_prob_model(x) + (1. - t) * log_prob(params["vd"], z))
  return potential


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

  def sampler(rng, x_0=None, xd_0=None, aux_0=None):
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

    # TODO: x_0 is initiated to zero in these underdamped solvers?
    if x_0 is None and xd_0 is None:
      if inner_solver:
        x, xd = inner_solver.prior(step_rng, shape)
      else:
        x, xd = outer_solver.prior(step_rng, shape)
    else:
      assert(x_0.shape==shape)
      if x_0 is None: raise ValueError("You must supply the complete state. x_0 was {}, expected array".format(None))
      if xd_0 is None: raise ValueError("You must supply the complete state. xd_0 was {}, expected array".format(None))
      x = x_0
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
        aux = inner_solver.init_aux(x)
      else:
        aux = outer_solver.init_aux(x)
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
