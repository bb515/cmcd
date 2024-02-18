import jax
import jax.numpy as jnp
import jax.random as random
from jax.lax import scan
from jax import jit, grad, value_and_grad
import numpyro.distributions as npdist
from functools import partial

from diffusionjax.utils import get_times


def compute_ratio(seed, params_flat, unflatten, params_fixed, log_prob, Solver):
  # TODO: this is like a sampler, but keeps parameters in one place, I think that's a good idea.
  # but it would be weird to train the beta schedule I think - but maybe that's the point of the work
  params_train, params_notrain = unflatten(params_flat)
  params_notrain = jax.lax.stop_gradient(params_notrain)
  params = {**params_train, **params_notrain}  # Gets all parameters in single place
  dim, num_steps, lfsteps = params_fixed

  # Need to reinitiate solver for every score
  ts = get_times(num_steps)
  beta = get_linear_beta_schedule()

  # TODO: params need to be correctly defined
  auxilliary_process_score = get_score(sde, model, params, score_scaling)
  outer_solver = Solver(base_process_score, auxilliary_process_score, beta, ts)

  rng, sample_rng = random.split(rng, 2)
  sampler = get_sampler(shape, outer_solver, denoise=False)
  x, x_mean, aux = sampler(sample_rng)

  # get sampler

  # Other fixed parameters
  params_fixed = (dim, nbridges, lfsteps)
  params_flat, unflatten = ravel_pytree((params_train, params_notrain))
  if nbridges >= 1:
      gridref_y = np.cumsum(params["mgridref_y"]) / np.sum(params["mgridref_y"])
      print(gridref_y)
      gridref_y = np.concatenate([np.array([0.0]), gridref_y])
      betas = np.interp(params["target_x"], params["gridref_x"], gridref_y)

  rng_key_gen = jax.random.PRNGKey(seed)

  rng_key, rng_key_gen = jax.random.split(rng_key_gen)
  z = vd.sample_rep(rng_key, params["vd"])
  w = -vd.log_prob(params["vd"], z)

  # Evolve UHA and update weight
  delta_H = np.array([0.0])
  if nbridges >= 1:
      rng_key, rng_key_gen = jax.random.split(rng_key_gen)
      z, w_mom, delta_H = ais_utils.evolve(
          z, betas, params, rng_key, params_fixed, log_prob
      )
      w += w_mom

  # Update weight with final model evaluation
  w = w + log_prob(z)
  delta_H = np.max(np.abs(delta_H))
  # delta_H = np.mean(np.abs(delta_H))
  return -1.0 * w, (z, delta_H)


def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob):
    ratios, (z, _) = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob
    )
    return ratios.mean(), (ratios, z)


def get_loss(sde, solver, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True):
  """Create a loss function for score matching training.
  Args:
    sde: Instantiation of a valid SDE class.
    solver: Instantiation of a valid Solver class.
    model: A valid flax neural network `:class:flax.linen.Module` class.
    score_scaling: Bool, set to `True` if learning a score scaled by the marginal standard deviation.
    likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    reduce_mean: Bool, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
    pointwise_t: Bool, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.

  Returns:
    A loss function that can be used for score matching training.
  """
  sampler = get_sampler(shape, outer_solver, inner_solver, denoise, stack_samples=False, inverse_scaler=None)
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
  def loss(params, rng, data):
    # Solver probably involves sampling forwards and backwards.
    rng, step_rng = random.split(rng)
    ts = random.uniform(step_rng, (data.shape[0],), minval=solver.ts[0], maxval=solver.t1)
    auxilliary_score = get_score(sde, model, params, score_scaling)
    compute_bound(auxilliary_score)


    e = errors(ts, sde, score, rng, data, likelihood_weighting)
    losses = e**2
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    if likelihood_weighting:
      g2 = sde.sde(jnp.zeros_like(data), ts)[1]**2
      losses = losses * g2
    return jnp.mean(losses)
  return loss


@partial(jit, static_argnums=[4])
def update_step(params, rng, batch, opt_state, loss):
  """
  Takes the gradient of the loss function and updates the model weights (params) using it.
  Args:
      params: the current weights of the model
      rng: random number generator from jax
      batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
      opt_state: the internal state of the optimizer
      loss: A loss function that can be used for score matching training.
  Returns:
      The value of the loss function (for metrics), the new params and the new optimizer states function (for metrics),
      the new params and the new optimizer state.
  """
  val, grads = value_and_grad(loss)(params, rng, batch)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return val, params, opt_state


def retrain_nn(
    info,
    update_step, num_epochs, step_rng, samples, params,
    opt_state, loss, batch_size=5):

  # TODO: assume optimizer has been initiated and is in update_step
  # Trying to replicate their training procedure before putting it in my own thing

  # have defined opt_state, ema_params, losses, and a looper
  # They using a cluster?

  train_size = samples.shape[0]
  batch_size = min(train_size, batch_size)
  steps_per_epoch = train_size // batch_size
  mean_losses = jnp.zeros((num_epochs, 1))
  for i in range(num_epochs):
    rng, step_rng = random.split(step_rng)
    # TODO: Needs to handle special case of diverging parameters
    # TODO: plot samples every 1% of training steps as check
    # TODO: consider using exponential_moving_average
    # TODO: log scalar metrics for every 1000 training steps using wandb


    perms = random.permutation(step_rng, train_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    losses = jnp.zeros((jnp.shape(perms)[0], 1))
    for j, perm in enumerate(perms):
      batch = samples[perm, :]
      rng, step_rng = random.split(rng)
      loss_eval, params, opt_state = update_step(params, step_rng, batch, opt_state, loss)
      losses = losses.at[j].set(loss_eval)
    mean_loss = jnp.mean(losses, axis=0)
    mean_losses = mean_losses.at[i].set(mean_loss)
    if i % 10 == 0:
      print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
  return params, opt_state, mean_losses


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


def ulangevin(log_prob, x, t):
  drift =  -1. * (beta * log_prob(x) + (1.0))
  diffusion = jnp.ones(x.shape) * jnp.sqrt(2)
  return drift, diffusion


def get_simple_annealed(log_prob, log_prob_variational, params, x, beta, clip=None):
  """Initiate a base process score."""
  # TODO: why does this resemble annealing and not a diffusion?
  # TODO: I don't understand why they have made this functional form requirement for gradU? I guess to make the experiments simple?
  # clipping is used for stability purposes
  if clip is not None:
    p = lambda x: log_prob(params, x)
    gp = grad(p)  # TODO: I thought jax grad acted on scalars... was this changed recently?
    u = lambda x: log_prob_model(x)
    gu = grad(u)
    def score(x, beta):
      return -1. * (beta *  jnp.clip(gu(x), -clip, clip) + (1. - beta) *  gp)
  else:
    def simple_annealed_negative_potential(x, beta):
      """Simple, tractable path of log densities based on series annealing."""
      return -1. * (beta * log_prob(x) + (1. - beta) * log_prob_variational(params, x))
    score = grad(simple_annealed_negative_potential)
  return score


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
  """For evaluating Markov transition kernels. Not the same as sampling,
  and so no computation should be wasted, and good to use a standardised library."""
  dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
  return dist.log_prob(x)


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
  params = {}
  import matplotlib.pyplot as plt

  # Everything related to betas
  ngridb = num_steps
  mgridref_y = jnp.ones(ngridb + 1) * 1.0
  params["gridref_x"] = jnp.linspace(0, 1, ngridb + 2)

  # BB: Does it not make sense to start this at 1. since density must be evaluated at 1. where the prior sample is initiated.
  # at the distribution that one starts in? i.e., shouldn't it instead be: # params_notrain["target_x"] = jnp.linspace(0, 1, num_steps + 1)[0:-1]
  params["target_x"] = jnp.linspace(0, 1, num_steps + 2)[1:-1]
  params["mgridref_y"] = mgridref_y

  # BB: I don't understand the purpose of gridref_y. Seems to be redundant since
  # y = f(x) is a linear relationship.
  # I guess that if a cosine schedule is used, then this would define that transform?
  gridref_y = jnp.cumsum(params["mgridref_y"]) / jnp.sum(params["mgridref_y"])
  gridref_y = jnp.concatenate([jnp.array([0.0]), gridref_y])

  ts = jnp.interp(params["target_x"], params["gridref_x"], gridref_y)
  dt = ts[1] - ts[0]

  # BB: Why not just have
  ts_alt = gridref_y[1:-1]
  dt_alt = ts_alt[1] - ts_alt[0]

  ts_bb, dt_bb = get_times(num_steps)
  print("dt: ", dt_bb, dt, dt_alt)
  print("len: ", ts.shape, ts_alt.shape, ts_bb.shape)
  print("maxmin", jnp.max(ts), jnp.min(ts))
  print(jnp.max(ts_bb), jnp.min(ts_bb))

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


def get_annealed_langevin(log_prob_model):
  def potential(x, t):
      return -1.0 * (
          t * log_prob_model(x) + (1.0 - t) * log_prob(params["vd"], z)
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
# get_betas()

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
