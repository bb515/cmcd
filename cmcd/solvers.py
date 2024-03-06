from diffusionjax.solvers import Solver, EulerMaruyama
from diffusionjax.utils import get_linear_beta_function, continuous_to_discrete, get_times, get_timestep
import numpyro.distributions as npdist
from cmcd.utils import log_prob_kernel, get_annealed_langevin, sample_rep, build
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax import vmap, random


MCD_SOLVERS = ["CMCDUD", "LeapfrogEA", "LeapfrogA", "LeapfrogE", "LeapfrogME", "LeapfrogACAIS", "CMCDOD", "VarCMCDOD", "MonteCarloDiffusion"]
UNDERDAMPED_SOLVERS = ["CMCDUD", "LeapfrogEA", "LeapfrogA", "LeapfrogE", "LeapfrogME", "LeapfrogACAIS", "UHA"]
OVERDAMPED_SOLVERS = ["CMCDOD", "VarCMCDOD", "MonteCarloDiffusion"]


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
  def __init__(self, params, shape, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None):
    """
    TODO: rename auxilliary_process_score into control_score or apply_score?
    Args:
      params:
      log_prob:
      base_process_score:
      auxilliary_process_score:
      beta:
      ts:
    """
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=0.1, beta_max=20.)
    self.params = params
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    self.auxilliary_process_score = auxilliary_process_score
    # TODO how to get shape at this point for solver
    # TODO is the scale definately 1.?
    # TODO: How come cannot use build(self.params) for dist_xd too?
    self.dist_x = build(self.params["vd"])
    # TODO: needs to be parallelized across batch size?
    self.dist_xd = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(self.params["vd"]["mean"]), scale=jnp.ones_like(self.params["vd"]["logdiag"])), 1)
    self.log_prob = log_prob

    # TODO: check that something like this works
    if base_process_score is None:
      base_process_potential = get_annealed_langevin(self.log_prob)
      base_process_score = jax.grad(base_process_potential)

    self.base_process_score = base_process_score

  def fin_aux(aux, xd):
    w = aux
    w = w + self.log_prob(xd)
    return w

  def init_aux(self, xd):
    w = - self.dist_xd.log_prob(xd)
    return w

  def prior(self, rng, shape):
    # TODO: Not sure about this!
    x_0 = sample_rep(rng, self.params["vd"])
    xd_0 = random.normal(rng, shape=shape)
    return x_0, xd_0

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
    xd_prime = fk_xd_mean + scale_f * random.normal(rng, shape=fk_xd_mean.shape)
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

  def init_params(params):
    eps = params["eps"]
    # NOTE: This is called eta_aux in the original code
    eta = params["gamma"] * eps
    scale = jnp.sqrt(2. * eta)
    scale_f = scale
    return scale, scale_f, eta, eps

  def forwards_kernel(self, eta, scale, eps, x, xd, t):
    fk_xd_mean = (1. - eta) * xd
    xd_prime = fk_xd_mean + scale * random.normal(rng, shape=fk_xd_mean.shape)
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
    xd_prime = fk_xd_mean + scale * random.normal(rng, shape=fk_xd_mean.shape)
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

  def init_params(params, t):
    i = get_timestep(t)
    # beta = self.discrete_betas(i)
    eta = params["eta"]
    eps = params["eps"]
    scale = jnp.sqrt(1. - eta**2)
    scale_f = scale
    return scale_f, scale, eta, eps

  def forwards_kernel(self, eta, scale, eps, x, xd, t):
    fk_xd_mean = eta * xd
    xd_prime = fk_xd_mean + scale * random.normal(rng, shape=fk_xd_mean.shape)
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


class LeapfrogACAIS(CMCDUD):
  """ACAIS"""

  def init_params(params, t):
    i = get_timestep(t)
    # eta = None
    # How to replace this with a time
    # This is actuallly a beta schedule multiplied by dt
    # eps = self.discrete_betas[i]
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


class CMCDOD(Solver):
  """
  Controlled Monte Carlo Diffusion Base Class. Overdamped (OD) SDE/Markov chain.
  need two subclasses:

  NOTE: this is the current implementation, MCD_CAIS_var_sn, evolve_overdamped_var_cais - that's the one with _eps_schedule, _cosine_eps_schedule
  TODO: this implementation, how different is it? MCD_CAIS_sn, evolve_overdamped_cais - also has _eps_schedule, _cosine_eps_schedule
  # TODO: I think this is for training params, but that should be done on external training loop
  that initiates the solver via supplying params. I think that's possible, need to test
  """
  def __init__(self, params, shape, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None):
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
    self.auxilliary_process_score = auxilliary_process_score
    self.dist_x = build(self.params["vd"])
    self.log_prob = log_prob

    # TODO: check that something like this works
    if base_process_score is None:
      base_process_potential = get_annealed_langevin(self.log_prob)
      base_process_score = jax.grad(base_process_potential)

    self.base_process_score = base_process_score

  def init_aux(self, x):
    # TODO: Check
    w = - self.dist_x.log_prob(x)
    return w

  def fin_aux(self, aux, x):
    """No final terms to add."""
    # TODO: Check
    w = aux
    w = w + self.log_prob(x)
    return w

  def init_params(self, t):
    i = get_timestep(t)
    eps = self.discrete_betas[i]
    scale = jnp.sqrt(2. * eps)
    return scale, eps

  def prior(self, rng, shape):
    # TODO: not sure about this!
    x_0 = sample_rep(rng, self.params["vd"])
    return x_0

  def evaluate_kernel_densities(self, x, x_new, fk_mean, bk_mean, scale):
    # Evaluate kernels
    # TODO: for overdamped, log prob kernel is not necessarily normal!
    dist = npdist.Independent(npdist.Normal(loc=fk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x_new)
    dist = npdist.Independent(npdist.Normal(loc=bk_mean, scale=scale), 1)
    bk_log_prob = dist.log_prob(x)
    return bk_log_prob, fk_log_prob

  def forwards_kernel(self, eps, scale, x, t):
    fk_mean = x - eps * self.base_process_score(x, t) - eps * self.auxilliary_process_score(self.params["sn"], x, t)
    x_new = x_mean + scale * random.normal(rng, x.shape)
    return x_new, fk_mean

  def backwards_kernel(self, eta, x, t):
    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - eps * self.base_process_score(x_new, t)
    else:
      bk_mean = (x_new
                - eps * self.base_process_score(x_new, t)
                + eps * self.auxilliary_process_score(self.params["sn"], x_new, t_next))
    return bk_mean

  def update(self, rng, x, t, aux):
    """NOTE: Uses a cosine_sq or linear beta schedule."""
    w = aux
    rng, step_rng = random.split(rng)
    scale, eps = self.init(self.params, t)
    x_new, fk_mean = self.forwards_kernel(eps, scale, x, t)
    bk_mean = self.backwards_kernel(eta, x, t)

    # NOTE: no stop gradient here.
    bk_log_prob, fk_log_prob = self.evaluate_kernel_densities(x, x_new, fk_mean, bk_mean, scale)

    # Update weight and return
    w += bk_log_prob - fk_log_prob
    aux = w
    return x_new, fk_mean, aux


class VarCMCDOD(CMCDOD):
  """"""

  def forwards_kernel(self, eps, scale, x, t):
    # NOTE: This is not DDPM, in DDPM there is a scaling by sqrt(1. - beta)
    x = stop_gradient(x)
    fk_mean = x - eps * self.base_process_score(x, t) - eps * self.auxilliary_process_score(self.params["sn"], x, t)
    x_new = x_mean + scale * random.normal(rng, x.shape)
    x_new = stop_gradient(x_new)
    return x_new, fk_mean

  def backwards_kernel(self, eta, x, t):
    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - eps * self.base_process_score(x_new, t)
    else:
      # TODO: is it t_next or t_prev?
      bk_mean = (x_new
                 - eps * self.base_process_score(x_new, t)
                 + eps * self.auxilliary_process_score(self.params["sn"], x_new, t_next))
    return bk_mean


class MonteCarloDiffusion(Solver):
  """
  # TODO: Shouldn't this inherit CMCDOD
  Monte Carlo Diffusion. Overdamped solver.
  from mc_ula_sn which is evolve_overdamped_orig
  """
  def __init__(self, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None):
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
    self.auxilliary_process_score = auxilliary_process_score

    # TODO: check that something like this works
    if base_process_score is None:
      base_process_potential = get_annealed_langevin(self.log_prob)
      base_process_score = jax.grad(base_process_potential)

    self.base_process_score = base_process_score

  def init_aux(self, x):
    w = 0.
    return w

  def prior(self, rng, shape):
    x_0 = sample_rep(rng, self.params["vd"])
    return x_0

  def init_params(self, t):
    i = get_timestep(t)
    eps = self.discrete_betas[i]
    scale = jnp.sqrt(2. * eps)
    return scale, eps

  def forwards_kernel(self, eps, scale, x, t):
    fk_mean = x - eps * self.base_process_score(x, t)
    x_new = x_mean + scale * random.normal(rng, x.shape)
    return x_new, fk_mean

  def backwards_kernel(self, eta, x, t):
    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - beta * self.base_process_score(x_new, t)
    else:
      bk_mean = (x_new
                - beta * self.base_process_score(x_new, t))
    return bk_mean

  def evaluate_kernel_densities(self, x, x_new, fk_mean, bk_mean, scale):
    # Evaluate kernels
    dist = npdist.Independent(npdist.Normal(loc=fk_mean, scale=scale), 1)
    fk_log_prob = dist.log_prob(x_new)
    dist = npdist.Independent(npdist.Normal(loc=bk_mean, scale=scale), 1)
    bk_log_prob = dist.log_prob(x)
    return bk_log_prob, fk_log_prob

  def final_aux(self, aux, shape):
    """No final terms to add."""
    return aux

  def update(self, rng, x, t, aux):
    # TODO: needs checking
    w = aux

    # Forward kernel
    rng, step_rng = random.split(rng)
    eps, scale = self.init_params(t)
    x_new, fk_mean = self.forwards_kernel(eps, scale, x, t)
    bk_mean = self.backwards_kernel(eta, x, t)

    # Evaluate kernels
    bk_log_prob, fk_log_prob = self.evaluate_kernel_densities(x, x_new, fk_mean, bk_mean, scale)

    # Update weight and return
    w += bk_log_prob - fk_log_prob
    aux = w
    return x, x_mean, aux


class UHA(CMCDUD):
  """
  This is just a method to compare against for the purposes of the paper.

  Need to know the base process value and potential. But this cannot be used with score estimate since score is not necessarily gradient of a potential.
  TODO: Not sure how it would differ from AISUDLangevin or other UD methods.
  NOTE: This implements  src/ais_utils.py the second-order momentum based version of Langevin dynamics (Uncorrected Hamiltonian Annealing), known as UHA.
  It implements leapfrog step to conserve the Hamiltonian (momentum equation). The version propose by Geffner et al used score networks (so presumably would need a network instead of a negative log potential?) along with 2nd order dynamics,
  but they implement it slightly differently in mcd_under_lp_a.
  TODO: compare to mcd_under_lp_a

  TODO: Annealed importance sampled Underdamped Langevin seems to keep scale of the state space the same, since x is not rescaled. Not sure it's what I want to implement first since I can't compare it to any other implementation
  From Thin et al. https://proceedings.mlr.press/v139/thin21a/thin21a.pdf
  Annealed Importance Sampling using Underdamped Langevin Markov transition
  kernels Markov chain.
  """
  def __init__(self, params, shape, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None, num_leapfrog_steps=None):
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
    super().__init__(params, shape, log_prob, base_process_score, auxilliary_process_score, beta, ts)
    self.num_leapfrog_steps = num_leapfrog_steps
    self.inner_ts = jnp.arange(self.num_leapfrog_steps - 1)

  def fin_aux(aux, xd):
    w = aux
    # NOTE: this is not done for ais_utils
    # w = aux
    # w = w + self.log_prob(xd)
    return w

  def init_aux(self, xd):
    w = 0
    # # NOTE: this is not done for ais_utils
    # w = - self.dist_xd.log_prob(xd)
    return w

  def prior(self, rng, shape):
    # TODO: Not sure about this!
    x_0 = sample_rep(rng, self.params["vd"])
    xd_0 = random.normal(rng, shape=shape)
    return x_0, xd_0

  def init_params(t):
    eps = self.params["eps"]
    eta = self.params["eta"]
    md = self.params["md"]
    # eta_aux = params["gamma"] * params["eps"]
    # scale = jnp.sqrt(2. * eta_aux)
    scale = jnp.sqrt(1.0 - eta**2) * jnp.exp(md)
    scale_f = scale
    return scale, scale_f, eta, eps, md

  def negative_potential_momentum(xd):
    # find the potential of the momentum, which is normally distributed.
    d_x = xd.shape[0]
    dist = npdist.Independent(np.dist.Normal(loc=jnp.zeros(d_x), scale=jnp.exp(self.params["md"])), 1)
    return -1. * dist.log_prob(xd)

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

  def update(self, rng, x, xd, t, aux):

    def _init(x, xd, rng):
      w = aux
      rng, step_rng = random.split(rng)
      xd = random.normal()
      # Evolve system
      rng, step_rng = random.split(rng)
      aux = w
      x, xd, aux, delta_H = jax.lax.scan(update, aux, ts)
      w = aux
      return x, w, delta_H

    w = aux
    scale, scale_f, eta, eps, md = init_params(t)

    # Half step for momentum
    negative_potential_x_final, grad_negative_potential_x = jax.value_and_grad(self.negative_potential)(x, t)
    xd = xd - eps * grad_negative_potential_x * .5
    negative_potential_xd_final = self.negative_potential_momentum(xd)
    # Re-sample momemtum
    xd = eta * xd_prev + scale_f * random.normal(rng, xd.shape)
    # Simulate dynamics
    # Half step for momentum, U is the 2nd order potential
    negative_potential_x_init, grad_negative_potential_x = jax.value_and_grad(self.negative_potential)(x, t)
    xd = xd - eps * .5 * grad_negative_potential_x
    # Full step for x
    negative_potential_xd_init, grad_negative_potential_xd = jax.value_and_grad(self.negative_potential_momentum)(xd)
    x = x + eps * grad_negative_potential_xd

    delta_H_init = negative_potential_x_init + negative_potential_xd_init
    # NOTE: I have used different way of calculating delta_H, offset by one half step. Although this can easily be fixed.
    delta_H = delta_H_init - negative_potential_x_final - negative_potential_xd_final

    w = w + log_prob(xd_new, md) - log_prob(xd, md)
    aux = w

    # Inner solver goes here
    # Alternate full steps
    # This is an inner solver but it can't be defined within an inner solver update inside a sampler?
    if num_leap_frog_steps > 1:
      (rng, x, xd, t) = scan(self.inner_step, (step_rng, x, xd, t), self.inner_ts)[0]

    return x, xd, aux, delta_H
