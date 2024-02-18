from diffusionjax.solvers import Solver, EulerMaruyama
from diffusionjax.utils import get_linear_beta_function, continuous_to_discrete, get_times, get_timestep
import numpyro.distributions as npdist
from cmcd.utils import log_prob_kernel, get_simple_annealed
import jax.numpy as jnp
from jax.lax import stop_gradient


class AISUDLangevin(Solver):
  """
  TODO: Assume that this is ais_utils, since it is Anealed Improtance Sampling with Uncorrected Underdamped Langevin.
  TODO: Doesn't actually look like ais_utils.
  TODO: Annealed importance sampled Underdamped Langevin seems to keep scale of the state space the same, since x is not rescaled. Not sure it's what I want to implement first since I can't compare it to any other implementation
  From Thin et al. https://proceedings.mlr.press/v139/thin21a/thin21a.pdf
  Annealed Importance Sampling using Underdamped Langevin Markov transition
  kernels Markov chain."""

  def __init__(self, base_process_score, auxilliary_process_score=None, beta=None, ts=None, full_score_network=True):
    """
    Args:
        score: grad_{x}(log p_{x, t})
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
    """
    # TODO: if this is the case, can just parent DDPM.
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=0.1, beta_max=20.)
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    # self.alphas = 1. - self.discrete_betas
    # self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    # self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    # self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
    # self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
    # self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
    # self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1. - self.alphas_cumprod_prev)
    self.base_process_score = base_process_score
    self.auxilliary_process_score = auxilliary_process_score
    self.gamma = gamma
    self.use_score_network = use_score_network
    self.full_score_network = full_score_network

  def init_aux(self, rng, shape):
    xd = random.normal(rng, shape)
    dist = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(xd), scale=1.), 1)
    # Add initial momentum term to w_0
    w = 0. - dist.log_prob(xd)
    return xd, w

  def prior(self, rng, shape):
    return random.normal(rng, shape)

  def final_update(self, aux, shape):
    """Add final momentum term to w. TODO: define this in solver base class? TODO: just as a function of aux."""
    xd, w = aux
    dist = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(xd), scale=1.), 1)
    # Add initial momentum term to w_0
    w = w + dist.log_prob(xd)
    return x, xd, aux

  def update(self, rng, x, xd, t, aux):

    def _init():
      # TODO: this may be confused as I'm not sure what this class is for.
      rng, step_rng = random.split(rng)
      xd = md.sample
      dim = xd.shape
      # (rng, params["eta"], None, params["md"])
      xd = jnp.exp(params["md"]) * random.normal(rng, (dim,))
      # else:
      #   xd = eta * prev + jnp.sqrt(1. - eta**2) * xd_indep

      # Evolve system
      w = 0.
      aux = w
      rng, step_rng = jax.random.split(rng)
      (rng, x, _, aux), delta_H = jax.lax.scan(evolve, aux, np.arange(num_steps))
      w = aux
      return x, w, delta_H

    def init(params, t):
      eps = params
      timestep = get_timestep(t)
      eps = self.discrete_betas[timestep]
      # eta_aux = params["gamma"] * params["eps"]
      eta_aux = self.gamma * eps
      scale = jnp.sqrt(2.0 * eta_aux)
      scale_f = scale
      return scale, scale_f, eps, eta_aux

    # TODO: needs checking
    # eps is dt, need to replace with it after all is done.
    # xd is a velocity (or, momentum) state, since this is a second order (Underdamped) solver
    w = aux
    # forward kernel
    # gamma is probably a timestep epsilon idk. It's not, I think it is a damping parameter
    # assume `eps` is dt... TODO: it's not. eps is discrete_beta
    # eps = self.dt

    # assume gamma is beta_max - probably wrong, but is doesn't appear in CAIS
    # gamma = self.beta_max
    # gamma doesn't seem to change so maybe it is a max beta
    scale, scale_f, eps, eta_aux = init(params, t)
    forward_kernel_mean = xd * (1. - beta_aux)
    rng, step_rng = jax.random.split(rng)

    z = random.normal(rng, x.shape)
    xd_prime = fk_xd_mean + scale * z
    xd_prime_prime = xd_prime - beta * 0.5 * self.base_process_score(x, t)
    x_new = x + beta * xd_prime_prime
    x_new_mean = x + beta * (xd_prime_prime - scale * z)
    xd_new = xd_prime_prime - beta * 0.5 * self.base_process_score(x_new, t)  # TODO: surely there is a way to evaluate this score network on the next iteration, but would have to save another state

    # so can save half on compute, or maybe the idea is that this is cheap compared to other computation.

    # Don't want to sample kernel here, want to do posterior step? Nah just write all at once, but get mean and std form a kernel method.
    # Backwards kernel
    # ULA does not use sn, whereas MCD does. LDVI uses sn, CAIS uses sn, 2nd order CAIS does not use sn. So maybe its methods that use a second learned
    # function, remember the a and b learned drift functions.
    if not self.use_score_network:  # TODO: what is sn? Probably something to do with grad ln phi, equation 25 in the ICLR submission
      # sn stands for "score network" and it is about whether a score network is needed in the method or not. Training probably requires sampling
      # forward and backwards through paths to evaluate stochastic integrals that are the lower bound on divergence between two path measures
      # sometimes it seems that the score network is fixed
      bk_xd_mean = xd_prime * (1.0 - eta_aux)
    else:
      if not self.full_score_network:
        # No real reason for this, just to try it out
        bk_xd_mean = xd_prime * (1.0 - eta_aux) + 2. * eta_aux * self.score_network(sn, z, i)
      else:
        # Score network received both z and the value of the forward transition kernel sample, it should need both because ??
        input_sn = jnp.concatenate([x, xd_prime])
        bk_xd_mean = xd_prime * (1.0 - eta_aux) + 2. * eta_aux * apply_fun_sn(sn, input_sn, i)

    # Evaluate kernels
    dist = npdist.Independent(npdist.Normal(loc=fk_xd_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(xd_prime)
    dist = npdist.Independent(npdist.Normal(loc=bk_xd_mean, scale=scale), 1)
    bk_log_prob = log_prob_kernel(xd, bk_log_prob, scale)

    # Update weight and return
    w += bk_log_prob - fk_log_prob

    rng, step_rng = jax.random.split(rng)
    aux = xd_new, w
    return x_new, xd_new, aux

    # x_mean, std = self.posterior(x, t)
    # z = random.normal(rng, x.shape)
    # x = x_mean + batch_mul(std, z)
    # return x, x_mean, aux

class CMCDUD(Solver):
  """
  Controlled Monte Carlo Diffusion Base Class. Underdamped (UD) SDE/Markov chain.
  evolve_underdamped_lp_a_cais (mcd_under_lp_a_cais) DONE
  evolve_underdamped_lp_a (mcd_under_lp_a) DONE
  evolve_underdamped_lp_e (mcd_under_lp_e) DONE
  evolve_underdamped_lp_ea (mcd_under_lp_ea) DONE
  NOTE: Their solver will scan over an arange(num_steps), going up in increment.
  """
  # expect there to be a damping term, and extended state space [x, xd].
  # It's the parallel, not series, combination of the scores which is a diffusion
  # They have, with their beta = my_alpha_t = 1 - my_v_t
  # Local information, by differentiating wrt p_0
  # - v_t * I^{-1} x - alpha_t C^{-1} x
  # I have global information, by integrating over p_0
  # - (v_t * I + alpha_t * C)^{-1} x
  def __init__(self, params, base_process_score, auxilliary_process_score=None, beta=None, ts=None, gamma=10., clip=1e2):
    """
    Args:
        score: grad_{x}(log p_{x, t})
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
    """
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=0.1, beta_max=20.)
    self.params = params
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    self.score = score
    self.auxilliary_process_score = auxilliary_process_score
    self.gamma = gamma
    self.base_process_score = base_process_score
    # TODO how to get shape at this point for solver
    self.dist = npdist.Independent(npdist.Normal(loc=jnp.zeros(xd.shape), scale=1.), 1)

  def fin_aux(aux, xd):
    w = aux
    w = w + self.dist.log_prob(xd)
    return w

  def init_aux(xd):
    w = 0.
    w = w - self.dist.log_prob(xd)
    return w

  def prior(rng, shape):
    return random.normal(rng, shape=(shape))

  def evaluate_kernel_densities(xd, xd_prime, fk_xd_mean, bk_xd_mean, scale, scale_f):
    dist = npdist.Independent(npdist.Normal(loc=fk_xd_mean, scale=scale_f), 1)
    fk_log_prob = dist.log_prob(xd_prime)
    dist = npdist.Independent(npdist.Normal(loc=bk_xd_mean, scale=scale), 1)
    bk_log_prob = log_prob_kernel(xd, bk_log_prob, scale)
    return fk_log_prob, bk_log_prob

  def update(self, rng, x, xd, t, aux):
    w = aux
    scale_f, scale, eta, eps = self.init_params(params, t)
    x_new, xd_new, xd_prime, fk_xd_mean = self.forward_kernel(eta, scale_f, eps, x, xd, t)
    bk_xd_mean = self.backwards_kernel(eta, x, xd_prime, t)
    bk_log_prob, fk_log_prob = self.evaluate_kernel_densities(xd, xd_prime, fk_xd_mean, bk_xd_mean, scale, scale_f)

    # Update weight
    w += bk_log_prob - fk_log_prob
    aux = w
    return x_new, xd_new, aux


class LeapfrogEA(CMCDUD):
  """EA"""

  def init_params(params):
    eps = params["eps"]
    eta = jnp.exp(-1. * params["gamma"] * eps)  # so that these parameters can be negative?
    eta_aux = params["gamma"] * eps  # NOTE: this better be close to zero
    scale = jnp.sqrt(2. * eta_aux)
    scale_f = jnp.sqrt(1. - eta**2)
    return scale_f, scale, eta_aux, eps

  def forwards_kernel(self, eta, scale, eps, x, xd, t):
    # Forward kernel
    fk_xd_mean = eta * xd
    xd_prime = fk_xd_mean + scale_f * jax.random.normal(rng, shape=fk_xd_mean.shape)
    xd_prime_prime = xd_prime - eps * self.base_process_score(x, t) * .5
    x_new = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(x, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backwards_kernel(self, eta, x, xd_prime, t):
    # Backwards kernel
    if not self.auxilliary_process_score:
      bk_xd_mean = (1. - eta) * xd_prime
    else:
      if not full_sn:
        bk_xd_mean = (1. - eta) * xd_prime + 2. * eta * self.auxilliary_process_score(params, x, t)
      else:
        input_sn = np.concatenate([x, xd_prime])
        bk_xd_mean = (1. - eta) * xd_prime + 2. * eta * self.auxilliary_process_score(params, input_sn, t)
    return bk_xd_mean


class LeapfrogA(CMCDUD):
  """A"""

  def init_lp_a(params):
    eps = params["eps"]
    # NOTE: This is called eta_aux in the original code
    eta = params["gamma"] * eps
    scale = jnp.sqrt(2. * eta)
    scale_f = scale
    return scale, scale_f, eta, eps

  def forwards_kernel(self, eta, scale, eps, x, xd, t):
    fk_xd_mean = (1. - eta) * xd
    xd_prime = fk_xd_mean + scale * jax.random.normal(rng, shape=fk_xd_mean.shape)
    xd_prime_prime = xd_prime - eps * self.base_process_score(x, t) * .5
    x_new = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(x, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backwards_kernel(self, eta, xd_prime, x, t):
    if self.auxilliary_process_score is None:
      bk_xd_mean = (1. - eta) * xd_prime
    else:
      if not self.full_sn:
        bk_xd_mean = (1. - eta) * xd_prime + 2 * eta * self.auxilliary_process_score(params["sn"], x, t)
      else:
        input_sn = jnp.concatenate([z, rho_prime])
        bk_xd_mean = xd_prime * (1. - eta) + 2 * eta * self.auxilliary_process_score(params["sn"], x, t)
    return bk_xd_mean



class LeapfrogE(CMCDUD):
  """E"""

  def init_params(self, params, t):
    eps = params["eps"]
    eta = params["eta"]
    scale = jnp.sqrt(1. - eta**2)
    scale_f = scale
    return scale, scale_f, eta, eps

  def forwards_kernel(self, eta, scale, eps, x, xd, t):
    # Forwards kernel
    fk_xd_mean = eta * xd
    xd_prime = fk_xd_mean + scale * jax.random.normal(rng, shape=fk_xd_mean.shape)
    xd_prime_prime = xd_prime - eps * self.base_process_score(x, t) * .5
    x = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(x, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backwards_kernel(self, eta, x, xd_prime, t):
    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_xd_mean = eta * xd_prime
    else:
      bk_xd_mean = eta * xd_prime + 2 * (1. - eta) * self.auxilliary_process_score(params["sn"], x, t)
    return bk_xd_mean


class LeapfrogME(CMCDUD):
  """ME"""

  def init_me_e(params, t):
    i = get_timestep(t)
    beta = self.discrete_betas(i)
    eta = params["eta"]
    eps = params["eps"]
    scale = jnp.sqrt(1. - eta**2)
    scale_f = scale
    return scale_f, scale, eta, eps

  def forwards_kernel(self, eta, scale, eps, x, xd, t):
    fk_xd_mean = eta * xd
    xd_prime = fk_xd_mean + scale * jax.random.normal(rng, shape=fk_xd_mean.shape)
    xd_new = xd_prime - eps * self.base_process_score(x, t)
    x_new = x + eps * xd_new
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backwards_kernel(self, eta, x, xd_prime, t):
    # Backwards kernel
    if not self.auxilliary_process_score:
      bk_xd_mean = eta * xd_prime
    else:
      bk_xd_mean = eta * xd_prime + 2. * self.auxilliary_process_score(params["sn"], x, t)
      # TODO: These were commented out.
      # bk_xd_mean = eta * xd_prime + 2. * (1. - eta) * self.auxilliary_process_score(params["sn"], x, t)
    return bk_xd_mean

  def update(self, rng, x, xd, t, aux):
    # Looks different to ea, due to bk_xd_mean, looks same as e
    w = aux
    scale_f, scale, eta, eps = self.init_me_e(params, t)

    # Forwards kernel
    x_new, xd_new, xd_prime, fk_xd_mean = self.forward_kernel(eta, scale, eps, x, xd, t)
    bk_xd_mean = self.backwards_kernel(eta, x, xd_prime, t)
    bk_log_prob, fk_log_prob = self.evaluate_kernel_densities(xd, xd_prime, fk_xd_mean, bk_xd_mean, scale, scale_f)

    # Update weight
    w += bk_log_prob - fk_log_prob
    aux = w
    return x_new, xd_new, aux


class LeapfrogACAIS(CMCDUD):
  """ACAIS"""

  def init_params(params, t):
    i = get_timestep(t)
    # eta = None
    # How to replace this with a time
    # This is actuallly a beta schedule multiplied by dt
    eps = self.discrete_betas[i]
    # eps = _cosine_eps_schedule(params["eps"], i)
    eta_aux = params["gamma"] * eps
    scale = jnp.sqrt(2. * eta_aux)
    scale_f = scale
    return scale, scale_f, eta_aux, eps

  def forwards_kernel(self, eta, scale, eps, x, xd, t):
    input_sn_old = jnp.concatenate([x, xd])
    fk_xd_mean = (1. - eta) * xd - 2. * eta * self.auxilliary_process_score(self.params["sn"], input_sn_old, t)

    # Leapfrog step
    xd_prime_prime = xd_prime - eps * self.base_process_score(x, t) * .5
    x_new = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(x_new, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backwards_kernel(self, eta, x, xd_prime, t):
    # TODO: Check these shouldn't be x_new and xd_new
    input_sn = jnp.concatenate([x, xd_prime])
    bk_xd_mean = (1. - eta) * xd_prime + 2. * eta * self.auxilliary_process_score(input_sn, t)
    return bk_xd_mean


class CMCDOD(ControlledMonteCarloDiffusion):
  """
  Controlled Monte Carlo Diffusion Base Class. Overdamped (OD) SDE/Markov chain.
  need two subclasses:

  NOTE: this is the current implementation, MCD_CAIS_var_sn, evolve_overdamped_var_cais - that's the one with _eps_schedule, _cosine_eps_schedule
  TODO: this implementation, how different is it? MCD_CAIS_sn, evolve_overdamped_cais - also has _eps_schedule, _cosine_eps_schedule
  # TODO: I think this is for training params, but that should be done on external training loop
  that initiates the solver via supplying params. I think that's possible, need to test
  """
  def __init__(self, params, base_process_score, auxilliary_process_score=None, beta=None, ts=None, gamma=10., clip=1e2):
    """
    Args:
        score: grad_{x}(log p_{x, t})
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
    """
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=0.1, beta_max=20.)
    self.params = params
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    self.score = score
    self.auxilliary_process_score = auxilliary_process_score
    self.gamma = gamma
    self.base_process_score = base_process_score
    # TODO how to get shape at this point for solver
    self.dist = npdist.Independent(npdist.Normal(loc=jnp.zeros(xd.shape), scale=1.), 1)

  def init_aux(self, rng, shape):
    w = 0.
    return w

  def prior(self, rng, shape):
    # TODO: Just check this
    return random.normal(rng, shape)

  def fin_aux(self, aux):
    """No final terms to add."""
    return aux

  def update_cais(self, rng, x, t, aux):
    w = aux
    def init(params, t):
      i = get_timestep(t)
      # eta = None
      # How to replace this with a time
      # This is actuallly a beta schedule multiplied by dt
      # their beta is actually be continuous times
      # their eps is a discrete beta schedule
      # eps is actually the largest discrete_beta,
      # half the variance of the kernel. Although in DDPM it's the variance of the kernel
      eps = self.discrete_betas[i]
      # eps = _cosine_eps_schedule(params["eps"], i)
      scale = jnp.sqrt(2. * eps)
      scale_f = scale
      return scale, scale_f, eps

    def _init():
      # sample initial momentum
      rng, step_rng = random.split(rng)
      w = 0.
      (rng, x, aux), _ = jax.lax.scan(evolve, aux, np.arange(num_steps))
      w = aux

    # Test
    w = aux

    rng, step_rng = random.split(rng)
    scale, scale_f, eps = init(params, t)
    fk_mean = x - eps * self.base_process_score(x, t) - eps * self.auxilliary_process_score(self.params["sn"], x, t)

    x_new = x_mean + scale * random.normal(rng, x.shape)
    # NOTE: no stop gradient here.

    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - eps * self.base_process_score(x_new, t)
    else:
      bk_mean = (x_new
                 - eps * self.base_process_score(x_new, t)
                 + eps * self.auxilliary_process_score(self.params["sn"], x_new, t_next))

    # Evaluate kernels
    dist = npdist.Independent(npdist.Normal(loc=fk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x_new)
    dist = npdist.Independent(npdist.Normal(loc=bk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x)

    # Update weight and return
    w += bk_log_prob - fk_log_prob
    aux = w
    return x_new, x_mean, aux

  def update_var_cais(self, rng, x, t, aux):
    """NOTE: Uses a cosine_sq or linear beta schedule."""
    w = aux

    def init(params, t):
      i = get_timestep(t)
      # eta = None
      # How to replace this with a time
      # This is actuallly a beta schedule multiplied by dt
      # their beta is actually be continuous times
      # their eps is a discrete beta schedule
      # eps is actually the largest discrete_beta,
      # half the variance of the kernel. Although in DDPM it's the variance of the kernel
      eps = self.discrete_betas[i]
      # eps = _cosine_eps_schedule(params["eps"], i)
      eta_aux = None
      scale = jnp.sqrt(2. * eps)
      scale_f = scale
      return scale, scale_f, eps, eta_aux

    def _init():
      # sample initial momentum
      rng, step_rng = random.split(rng)
      xd = random.normal(rng, shape=(xd.shape))

      dist = npdist.Independent(npdist.Normal(loc=jnp.zeros(xd.shape), scale=1.), 1)
      w = 0. - dist.log_prob(xd)

      # Evolve system
      aux = w
      rng, step_rng = jax.random.split(rng)
      (rng, x, xd, aux), _ = jax.lax.scan(evolve, aux, np.arange(num_steps))

      # Add final momentum term to w
      dist = npdist.Independent(npdist.Normal(loc=jnp.zeros(xd.shape), scale=1.), 1)
      w = w + dist.log_prob(xd)

    x = stop_gradient(x)
    rng, step_rng = random.split(rng)
    scale, scale_f, eps = init(self.params, t)

    # NOTE: This is not DDPM, in DDPM there is a scaling by sqrt(1. - beta)
    fk_mean = x - eps * self.base_process_score(x, t) - eps * self.auxilliary_process_score(self.params["sn"], x, t)

    x_new = x_mean + scale * random.normal(rng, x.shape)
    x_new = stop_gradient(x_new)

    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - eps * self.base_process_score(x_new, t)
    else:
      # TODO: is it t_next or t_prev?
      bk_mean = (x_new
                 - eps * self.base_process_score(x_new, t)
                 + eps * self.auxilliary_process_score(self.params, x_new, t_next))

    # Evaluate kernels
    dist = npdist.Independent(npdist.Normal(loc=fk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x_new)
    dist = npdist.Independent(npdist.Normal(loc=bk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x)

    # Update weight and return
    w += bk_log_prob - fk_log_prob
    aux = w
    return x_new, x_mean, aux


class MonteCarloDiffusion(Solver):
  """
  Monte Carlo Diffusion. Overdamped solver.
  from mc_ula_sn which is evolve_overdamped_orig
  """
  def __init__(self, base_process_score, auxilliary_process_score=None, beta=None, ts=None, gamma=10.0):
    """
    Args:
        score: grad_{x}(log p_{x, t})
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
    """
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=0.1, beta_max=20.)
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    self.base_process_score = base_process_score
    self.auxilliary_process_score = auxilliary_process_score
    self.gamma = gamma

  def init_aux(self, rng, shape):
    w = 0.
    return w

  def prior(self, rng, shape):
    # TODO: Just check this
    return random.normal(rng, shape)

  def final_update(self, aux, shape):
    """No final terms to add."""
    return aux

  def update(self, rng, x, t, aux):
    # TODO: needs checking
    w = aux
    timestep = get_timestep(t)
    beta = self.discrete_betas[timestep]  # NOTE: that they have discrete betas for this diffusion sampler and not the annealed Langevin sampler

    # Forward kernel
    fk_mean = x - beta * self.base_process_score(x, t)
    scale = jnp.sqrt(2. * beta)
    rng, step_rng = random.split(rng)
    z = random.normal(rng, x.shape)
    x_mean = fk_mean
    x_new = x_mean + scale * z

    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - beta * self.base_process_score(x_new, t)
    else:
      bk_mean = (x_new
                 - beta * self.base_process_score(x_new, t)
                 + beta * self.auxilliary_process_score(self.params, x_new, t))

    # Evaluate kernels
    dist = npdist.Independent(npdist.Normal(loc=fk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x_new)
    dist = npdist.Independent(npdist.Normal(loc=bk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x)

    # Update weight and return
    w += bk_log_prob - fk_log_prob
    aux = w
    return x, x_mean, aux


class AISUD_unknown(Solver):
  """
  This is just a method to compare against. For the purposes of the paper. No need for auxilliary process or whathaveyou.

  Need to know the base process value and potential. But this cannot be used with score estimate since score is not necessarily gradient of a potential.
  TODO: This is from src/ais_utils.py. Not sure how it would differ from AISUDLangevin or other UD methods.
  NOTE: This implements the second-order momentum based version of Langevin dynamics (Uncorrected Hamiltonian Annealing), known as UHA.
  It implements leapfrog step to conserve the Hamiltonian (momentum equation). The version propose by Geffner et al used score networks (so presumably would need a network instead of a negative log potential?) along with 2nd order dynamics,
  but they implement it slightly differently in mcd_under_lp_a.
  TODO: compare to mcd_under_lp_a

  TODO: Annealed importance sampled Underdamped Langevin seems to keep scale of the state space the same, since x is not rescaled. Not sure it's what I want to implement first since I can't compare it to any other implementation
  From Thin et al. https://proceedings.mlr.press/v139/thin21a/thin21a.pdf
  Annealed Importance Sampling using Underdamped Langevin Markov transition
  kernels Markov chain."""

  def __init__(self, params, negative_potential, num_leapfrog_steps=1, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None, full_score_network=True):
    """
    Args:
      # TODO: can the params we differentiate objective wrt be passed here and defined as arguments in normal way?
      negative_potential: ( - log p_(x, t) ) negative potential of the base process.
      base_process_score: TODO placeholder, probably not needed.
      aux_process_score: TODO placeholder, probably not needed.
      beta: TODO: placeholder, probably not needed.
      score: grad_{x}(log p_{x, t})
      model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
      eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
    """
    # TODO: if this is the case, can just parent DDPM.
    # TODO: change the params names, currently have them as
    # params["eps"], params["md"]
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=0.1, beta_max=20.)
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    # self.alphas = 1. - self.discrete_betas
    # self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    # self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    # self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
    # self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
    # self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
    # self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1. - self.alphas_cumprod_prev)

    self.params = params
    self.negative_potential = negative_potential
    self.num_leapfrog_steps = num_leapfrog_steps
    self.inner_ts = jnp.arange(self.num_leapfrog_steps - 1)

    # self.base_process_score = base_process_score
    # self.auxilliary_process_score = auxilliary_process_score
    # self.gamma = gamma
    # self.use_score_network = use_score_network
    # self.full_score_network = full_score_network

  def negative_potential_momentum(xd):
    # find the potential of the momentum, which is normally distributed.
    d_x = xd.shape[0]
    dist = npdist.Independent(np.dist.Normal(loc=jnp.zeros(d_x), scale=jnp.exp(self.params["md"])), 1)
    return -1. * dist.log_prob(xd)

  def init_aux(self, rng, shape):
    xd = random.normal(rng, shape)
    dist = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(xd), scale=1.), 1)
    # Add initial momentum term to w_0
    w = 0. - dist.log_prob(xd)
    return xd, w

  def prior(self, rng, shape):
    return random.normal(rng, shape)

  def final_update(self, aux, shape):
    """Add final momentum term to w. TODO: define this in solver base class? TODO: just as a function of aux."""
    xd, w = aux
    dist = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(xd), scale=1.), 1)
    # Add final momentum term to w
    w = w + dist.log_prob(xd)
    return x, xd, aux

  def inner_update(self, x, xd, t):
    eps = self.params["eps"]
    xd = xd - eps * jax.grad(self.negative_potential)(x, t)
    x = x + eps * jax.grad(self.negative_potential_momentum)(xd)
    return x, xd

  def inner_step(self, carry, i):
    x, xd, t = carry
    rng, step_rng = random.split(rng)
    x, xd = self.inner_update(x, xd, t)
    return (rng, x, xd, t), None

  def update(self, rng, x, t, aux):

    def init(params):
      eps = params["eps"]
      eta = params["eta"]
      md = params["md"]
      # eta_aux = params["gamma"] * params["eps"]
      scale = jnp.sqrt(2. * eta_aux)
      scale_f = scale
      return scale, scale_f, eta, eps, md

    # TODO: needs checking
    # rng is split outside of update method
    # eps is dt, need to replace with it after all is done.
    # xd is a velocity (or, momentum) parameter, since this is a second order (Underdamped) solver
    w = aux
    scale, scale_f, eta, eps, md = init(params)

    # forward kernel
    # assume `eps` is dt
    # eps = self.dt

    # Re-sample momemtum
    # TODO eta is discrete_beta. why squared? starts to look like DDPM
    # TODO eta is fixed constant? yes, eta is the damping constant for 2nd order dynamics.
    # Can one have an adaptive damping constant? seems like a good idea, but how to implement?
    # params["eps"] - there is no adaptive time stepping in this scheme.
    xd = beta * xd_prev + jnp.sqrt(1.0 - eta**2) * jnp.exp(md) * random.normal(rng, xd.shape)
    # Simulate dynamics
    # z_new, xd_new, delta_H
    # Half step for momentum, U is the 2nd order potential
    negative_potential_x_init, grad_negative_potential_x = jax.value_and_grad(self.negative_potential)(x, t)

    xd = xd - eps * .5 * grad_negative_potential_x
    # Full step for x
    negative_potential_xd_init, grad_negative_potential_xd = jax.value_and_grad(self.negative_potential_momentum)(xd)
    x = x + eps * grad_negative_potential_xd

    # Alternat full steps
    # This is an inner solver but it can't be defined within update?
    if num_leap_frog_steps > 1:
      (rng, x, xd, t) = scan(self.inner_step, (step_rng, x, xd, t), self.inner_ts)[0]

    # Half step for momentum
    negative_potential_x_final, grad_negative_potential_x = jax.value_and_grad(self.negative_potential)(x, t)
    xd = xd - eps * grad_negative_potential_x * .5
    negative_potential_xd_final = self.negative_potential_momentum(xd)

    delta_H = negative_potential_x_init - negative_potential_x_final + negative_potential_xd_init - negative_potential_xd_final

    return x, xd, delta_H

