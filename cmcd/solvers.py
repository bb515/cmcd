"""TODO: rename auxilliary_process_score into control_score or apply_score?

NOTE: These methods are not DDPM, in DDPM there is a scaling by sqrt(1. - beta)
NOTE:
  They have an annealing scheme that only uses local information, by differentiating wrt p_0
  - v_t * I^{-1} x - alpha_t C^{-1} x
  whereas diffusion has global information, by integrating over p_0 - (v_t * I + alpha_t * C)^{-1} x
 NOTE: Their solver will scan over an arange(num_steps), going up in increment.

"""
from abc import abstractmethod
from diffusionjax.solvers import Solver, EulerMaruyama
from diffusionjax.utils import get_linear_beta_function, continuous_to_discrete, get_times, get_timestep
import numpyro.distributions as npdist
from cmcd.utils import log_prob_kernel, get_annealed_langevin, sample_rep, build
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax import vmap, random, grad, value_and_grad


MCD_SOLVERS = ["LeapfrogEA", "LeapfrogA", "LeapfrogE", "LeapfrogME", "LeapfrogACAIS", "CMCDOD", "VarCMCDOD", "MonteCarloDiffusion"]
UNDERDAMPED_SOLVERS = ["LeapfrogEA", "LeapfrogA", "LeapfrogE", "LeapfrogME", "LeapfrogACAIS", "UHA"]
OVERDAMPED_SOLVERS = ["CMCDOD", "VarCMCDOD", "MonteCarloDiffusion"]


class CMCDUD(Solver):
  """
  Controlled Monte Carlo Diffusion abstract base class.

  For second-order underdamped (UD) dynamical system of SDE/Markov chain.
  """
  def __init__(self, params, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None):
    """
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
      base_process_score = grad(base_process_potential)

    self.base_process_score = base_process_score

  @abstractmethod
  def init_params(self, t):
    """
    Initialize the params.
    """
    # TODO: investigate keeping this stuff within diffusionjax framework?
    # get_timestep(t)
    # discrete_betas[i]
    # eps is a beta schedule multiplied by dt?
    # eps = self.discrete_betas[i]
    # eps = _cosine_eps_schedule(params["eps"], i)

  @abstractmethod
  def forward_kernel(self):
    """ """

  @abstractmethod
  def backward_kernel(self):
    """ """

  def fin_aux(self, aux, xd):
    w = aux
    w = w + self.log_prob(xd)
    return w

  def init_aux(self, xd):
    # NOTE: from mcdboundingmachine.py
    w = - self.dist_xd.log_prob(xd)
    # NOTE: from the individual evolve methods
    w = w - self.log_prob_kernel(xd, jnp.zeros_like(xd), 1.0)
    return w

  def prior(self, rng, shape):
    x_0 = sample_rep(rng, self.params["vd"])
    xd_0 = random.normal(rng, shape=shape)
    return x_0, xd_0

  def log_prob_kernel(self, xd, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(xd)

  def update(self, rng, x, xd, t, aux):
    w = aux
    scale_f, scale, eta, eps = self.init_params(t)
    x_new, xd_new, xd_prime, fk_xd_mean = self.forward_kernel(rng, eta, scale_f, eps, x, xd, t)
    bk_xd_mean = self.backward_kernel(eta, x_new, xd_prime, t)

    # Update weight
    fk_log_prob = self.log_prob_kernel(xd_prime, fk_xd_mean, scale_f)
    bk_log_prob = self.log_prob_kernel(xd, bk_xd_mean, scale)
    w += bk_log_prob - fk_log_prob
    aux = w
    return x_new, xd_new, aux


class LeapfrogEA(CMCDUD):
  """

  NOTE: implements from mcd_under_lp_ea import evolve_underdamped_lp_ea

  aka
    elif mode == "MCD_U_ea-lp-sn":

  evolve_underdamped_lp_ea (mcd_under_lp_ea)"""
  def init_params(self, t):
    eps = self.params["eps"]
    gamma = self.params["gamma"]
    eta = jnp.exp(-1. * self.params["gamma"] * eps)  # so that these parameters can be negative?
    eta_aux = self.params["gamma"] * eps  # NOTE: this better be close to zero
    scale = jnp.sqrt(2. * eta_aux)
    scale_f = jnp.sqrt(1. - eta**2)
    return scale_f, scale, eta_aux, eps

  def forward_kernel(self, rng, eta, scale_f, eps, x, xd, t):
    # Forward kernel
    fk_xd_mean = eta * xd
    xd_prime = fk_xd_mean + scale_f * random.normal(rng, shape=fk_xd_mean.shape)
    bps = self.base_process_score(self.params["vd"], x, t)
    xd_prime_prime = xd_prime - eps * self.base_process_score(self.params["vd"], x, t) * .5
    x_new = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(self.params["vd"], x, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backward_kernel(self, eta, x, xd_prime, t):
    # Backwards kernel
    if not self.auxilliary_process_score:
      bk_xd_mean = (1. - eta) * xd_prime
    else:
      if not full_sn:
        bk_xd_mean = (1. - eta) * xd_prime + 2. * eta * self.auxilliary_process_score(self.params["sn"], x, t)
      else:
        input_sn = np.concatenate([x, xd_prime])
        bk_xd_mean = (1. - eta) * xd_prime + 2. * eta * self.auxilliary_process_score(self.params["sn"], input_sn, t)
    return bk_xd_mean


class LeapfrogA(CMCDUD):
  """
  NOTE: LDVI uses MCD_U_a-lp-sn
  NOTE: implements from mcd_under_lp_a import evolve_underdamped_lp_a
  aka mode == "MCD_U_a-lp", "MCD_U_a-lp-sna", "MCD_U_a-lp-sn"
  """
  def init_params(self, t):
    eps = self.params["eps"]
    # NOTE: This is called eta_aux in the original code
    eta = self.params["gamma"] * eps
    scale = jnp.sqrt(2. * eta)
    scale_f = scale
    return scale, scale_f, eta, eps

  def forward_kernel(self, rng, eta, scale, eps, x, xd, t):
    fk_xd_mean = (1. - eta) * xd
    xd_prime = fk_xd_mean + scale * random.normal(rng, shape=fk_xd_mean.shape)
    xd_prime_prime = xd_prime - eps * self.base_process_score(self.params["vd"], x, t) * .5
    x_new = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(self.params["vd"], x, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backward_kernel(self, eta, xd_prime, x, t):
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
  """

  NOTE: implements from mcd_under_lp_e import evolve_underdamped_lp_e
  aka,
   elif mode == "MCD_U_e-lp":
   elif mode == "MCD_U_e-lp-sna":
  """
  def init_params(self, t):
    eps = self.params["eps"]
    eta = self.params["eta"]
    scale = jnp.sqrt(1. - eta**2)
    scale_f = scale
    return scale, scale_f, eta, eps

  def forward_kernel(self, rng, eta, scale, eps, x, xd, t):
    # Forwards kernel
    fk_xd_mean = eta * xd
    xd_prime = fk_xd_mean + scale * random.normal(rng, shape=fk_xd_mean.shape)
    xd_prime_prime = xd_prime - eps * self.base_process_score(self.params["vd"], x, t) * .5
    x_new = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(self.params["vd"], x, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backward_kernel(self, eta, x, xd_prime, t):
    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_xd_mean = eta * xd_prime
    else:
      bk_xd_mean = eta * xd_prime + 2 * (1. - eta) * self.auxilliary_process_score(params["sn"], x, t)
    return bk_xd_mean


class LeapfrogME(CMCDUD):
  """ME"""
  def init_params(self, t):
    eta = self.params["eta"]
    eps = self.params["eps"]
    scale = jnp.sqrt(1. - eta**2)
    scale_f = scale
    return scale_f, scale, eta, eps

  def forward_kernel(self, rng, eta, scale, eps, x, xd, t):
    fk_xd_mean = eta * xd
    xd_prime = fk_xd_mean + scale * random.normal(rng, shape=fk_xd_mean.shape)
    xd_new = xd_prime - eps * self.base_process_score(self.params["vd"], x, t)
    x_new = x + eps * xd_new
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backward_kernel(self, eta, x, xd_prime, t):
    # Backwards kernel
    if not self.auxilliary_process_score:
      bk_xd_mean = eta * xd_prime
    else:
      bk_xd_mean = eta * xd_prime + 2. * self.auxilliary_process_score(params["sn"], x, t)
      # TODO: These were commented out.
      # bk_xd_mean = eta * xd_prime + 2. * (1. - eta) * self.auxilliary_process_score(params["sn"], x, t)
    return bk_xd_mean


class LeapfrogACAIS(CMCDUD):
  """
  NOTE: implements from mcd_under_lp_a_cais import evolve_underdamped_lp_a_cais
  aka
    elif mode == "MCD_CAIS_UHA_sn"
  NOTE: 2nd order CMCD uses MCD_CAIS_UHA_sn
  """
  def init_params(self, t):
    # TODO: eps should be eps = _cosine_eps_schedule(params["eps"], i)
    eps = self.params["eps"]
    eta_aux = self.params["gamma"] * eps
    scale = jnp.sqrt(2. * eta_aux)
    scale_f = scale
    return scale, scale_f, eta_aux, eps

  def forward_kernel(self, rng, eta, scale, eps, x, xd, t):
    input_sn_old = jnp.concatenate([x, xd])
    i = get_timestep(t, 1.0, 0.0, 8)
    fk_xd_mean = (1. - eta) * xd - 2. * eta * self.auxilliary_process_score(self.params["sn"], input_sn_old, i)
    xd_prime  = fk_xd_mean + scale * random.normal(rng, x.shape)

    # Leapfrog step
    xd_prime_prime = xd_prime - eps * self.base_process_score(self.params["vd"], x, t) * .5
    x_new = x + eps * xd_prime_prime
    xd_new = xd_prime_prime - eps * self.base_process_score(self.params["vd"], x_new, t) * .5
    return x_new, xd_new, xd_prime, fk_xd_mean

  def backward_kernel(self, eta, x, xd_prime, t):
    # TODO: Check these shouldn't be x_new and xd_new
    input_sn = jnp.concatenate([x, xd_prime])
    i = get_timestep(t, 1.0, 0.0, 8)
    bk_xd_mean = (1. - eta) * xd_prime + 2. * eta * self.auxilliary_process_score(self.params["sn"], input_sn, i)
    return bk_xd_mean


class CMCDOD(Solver):
  """
  Controlled Monte Carlo Diffusion base class for Overdamped (OD) SDE/Markov chain.

  NOTE: implements from mcd_over_orig import evolve_overdamped_orig
  aka
  if mode == "MCD_ULA"
  elif mode == "MCD_ULA_sn"
  NOTE: also implements from mcd_cais import evolve_overdamped_cais
  aka
  elif mode == "MCD_CAIS_sn": TODO: use _eps_schedule, _cosine_eps_schedule in that case
  NOTE: CMCD uses MCD_CAIS_sn
  NOTE: ULA uses MCD_ULA
  NOTE: MCD uses MCD_ULA_sn
  """
  def __init__(self, params, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None):
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

    if base_process_score is None:
      base_process_potential = get_annealed_langevin(self.log_prob)
      base_process_score = grad(base_process_potential)

    self.base_process_score = base_process_score

  def init_aux(self, x):
    w = - self.dist_x.log_prob(x)
    return w

  def fin_aux(self, aux, x):
    w = aux
    w = w + self.log_prob(x)
    return w

  def init_params(self, t):
    # TODO: eps could also be consine_eps_echedule or linear_eps_schedule
    eps = self.params["eps"]
    scale = jnp.sqrt(2. * eps)
    return scale, eps

  def prior(self, rng, shape):
    # TODO: not sure about this!
    x_0 = sample_rep(rng, self.params["vd"])
    return x_0

  def log_prob_kernel(self, x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)

  def forward_kernel(self, rng, eps, scale, x, t):
    i = get_timestep(t, 1.0, 0.0, 8)
    fk_mean = x - eps * self.base_process_score(self.params["vd"], x, t) - eps * self.auxilliary_process_score(self.params["sn"], x, i)
    x_new = fk_mean + scale * random.normal(rng, x.shape)
    return x_new, fk_mean

  def backward_kernel(self, eps, x_new, t):
    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - eps * self.base_process_score(self.params["vd"], x_new, t)
    else:
      # TODO: is this getting an index i increasing in scan index?
      i = get_timestep(t, 1.0, 0.0, 8)
      i_next = i + 1
      bk_mean = (x_new
                - eps * self.base_process_score(self.params["vd"], x_new, t)
                + eps * self.auxilliary_process_score(self.params["sn"], x_new, i_next))
    return bk_mean

  def update(self, rng, x, t, aux):
    """NOTE: Uses a cosine_sq or linear beta schedule."""
    w = aux
    rng, step_rng = random.split(rng)
    scale, eps = self.init_params(t)
    x_new, fk_mean = self.forward_kernel(rng, eps, scale, x, t)
    bk_mean = self.backward_kernel(eps, x_new, t)

    # NOTE: no stop gradient here.
    fk_log_prob = self.log_prob_kernel(x_new, fk_mean, scale)
    bk_log_prob = self.log_prob_kernel(x, bk_mean, scale)

    # Update weight and return
    w += bk_log_prob - fk_log_prob
    aux = w
    return x_new, fk_mean, aux


class VarCMCDOD(CMCDOD):
  """
  evolve_overdamped_var_cais aka MCD_CAIS_var_sn aka mcd_cais_var
  NOTE: CMCD + VarGrad loss uses MCD_CAIS_var_sn

  TODO: use _eps_schedule, _cosine_eps_schedule
  """
  def forward_kernel(self, rng, eps, scale, x, t):
    i = get_timestep(t, 1.0, 0.0, 8)
    x = stop_gradient(x)
    fk_mean = x - eps * self.base_process_score(self.params["vd"], x, t) - eps * self.auxilliary_process_score(self.params["sn"], x, i)
    x_new = fk_mean + scale * random.normal(rng, x.shape)
    x_new = stop_gradient(x_new)
    return x_new, fk_mean


class MonteCarloDiffusion(CMCDOD):
  """
  NOTE: no parameters for this method, as a baseline
  Monte Carlo Diffusion. Overdamped solver.
  from mc_ula_sn which is evolve_overdamped_orig in mcd_over_orig
  """

  def forward_kernel(self, rng, eps, scale, x, t):
    fk_mean = x - eps * self.base_process_score(self.params["vd"], x, t)
    x_new = fk_mean + scale * random.normal(rng, x.shape)
    return x_new, fk_mean

  def backward_kernel(self, eps, x_new, t):
    # Backwards kernel
    if self.auxilliary_process_score is None:
      bk_mean = x_new - eps * self.base_process_score(self.params["vd"], x_new, t)
    else:
      i = get_timestep(t, 1.0, 0.0, 8)
      bk_mean = (x_new
                - eps * self.base_process_score(self.params["vd"], x_new, t)
                 + eps * self.auxilliary_process_score(self.params["sn"], x_new, i)
                 )
    return bk_mean


class UHA(Solver):
  """
  as a baseline, this is just a method to compare against for the purposes of the paper.

  NOTE: UHA uses UHA

  Need to know the base process value and potential. But this cannot be used with score estimate since score is not necessarily gradient of a potential.
  TODO: Not sure how it would differ from AISUDLangevin or other UD methods.
  NOTE: delta_H only appears in this solver.
  NOTE: This implements  src/ais_utils.py the second-order momentum based version of Langevin dynamics (Uncorrected Hamiltonian Annealing), known as UHA.
  It implements leapfrog step to conserve the Hamiltonian (momentum equation). The version propose by Geffner et al used score networks (so presumably would need a network instead of a negative log potential?) along with 2nd order dynamics,
  but they implement it slightly differently in mcd_under_lp_a.
  TODO: compare to mcd_under_lp_a

  TODO: Annealed importance sampled Underdamped Langevin seems to keep scale of the state space the same, since x is not rescaled. Not sure it's what I want to implement first since I can't compare it to any other implementation
  From Thin et al. https://proceedings.mlr.press/v139/thin21a/thin21a.pdf
  Annealed Importance Sampling using Underdamped Langevin Markov transition
  kernels Markov chain.
  """
  def __init__(self, params, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None, num_leapfrog_steps=None):
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

    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=0.1, beta_max=20.)
    self.params = params
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    self.auxilliary_process_score = auxilliary_process_score
    self.dist_x = build(self.params["vd"])
    self.dist_xd = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(self.params["vd"]["mean"]), scale=jnp.ones_like(self.params["vd"]["logdiag"])), 1)
    self.log_prob = log_prob
    self.num_leapfrog_steps = num_leapfrog_steps
    self.inner_ts = jnp.arange(self.num_leapfrog_steps - 1)
    self.base_process_potential = get_annealed_langevin(self.log_prob)
    self.base_process_value_and_grad = value_and_grad(self.base_process_potential, argnums=1)
    self.base_process_momentum_score = grad(self.base_process_momentum_potential)
    self.base_process_momentum_value_and_grad = value_and_grad(self.base_process_momentum_potential)

  def prior(self, rng, shape):
    x_0 = sample_rep(rng, self.params["vd"])
    xd_0 = random.normal(rng, shape=shape)
    return x_0, xd_0

  def init_aux(self, xd):
    # TODO: is there any way to remove delta_H from this?
    w = - self.dist_xd.log_prob(xd)
    delta_H = 0
    return w, delta_H

  def fin_aux(self, aux, xd):
    w, delta_H = aux
    w = w + self.log_prob(xd)
    return w, delta_H

  def init_params(self, t):
    eps = self.params["eps"]
    eta = self.params["eta"]
    md = self.params["md"]
    scale = jnp.sqrt(1.0 - eta**2) * jnp.exp(md)
    scale_f = scale
    return scale, scale_f, eta, eps, md

  def base_process_momentum_potential(self, xd):
    # find the potential of the momentum, which is normally distributed.
    dist = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(xd), scale=jnp.exp(self.params["md"])), 1)
    return -1. * dist.log_prob(xd)
  # def log_prob_momdist(xd, params):
  #     dist = npdist.Independent(npdist.Normal(loc=jnp.zeros_like(xd), scale=jnp.exp(params)), 1)
  #     return dist.log_prob(xd)

  # TODO: needed?
  def sample_momdist(rng_key, eta, prev, params):
      # Params is just an array with logscale parameters
      dim = params.shape[0]
      xd_indep = np.exp(params) * jax.random.normal(rng_key, params.shape)
      if prev is None:
        xd = xd_indep
      else:
        xd = eta * prev + np.sqrt(1.0 - eta**2) * xd_indep
      return xd

  def inner_update(self, x, xd, t):
    eps = self.params["eps"]
    # xd = xd - eps * grad(self.negative_potential)(x, t)
    xd = xd - eps * self.base_process_score(params["vd"], x, t)
    x = x + eps * self.base_process_momentum_score(xd)
    # x = x + eps * grad(self.negative_potential_momentum)(xd)
    return x, xd

  def inner_step(self, carry, i):
    x, xd, t = carry
    rng, step_rng = random.split(rng)
    x, xd = self.inner_update(x, xd, t)
    return (rng, x, xd, t), None

  def update(self, rng, x, xd, t, aux):

    # TODO: needed? what is this for... presumably if get_sampler needs to be different.
    def _init(x, xd, rng):
      w = aux
      rng, step_rng = random.split(rng)
      xd = random.normal()
      # Evolve system
      rng, step_rng = random.split(rng)
      aux = w
      x, xd, aux = jax.lax.scan(update, aux, ts)
      w, delta_H = aux
      return x, aux

    w, delta_H = aux
    scale, scale_f, eta, eps, md = self.init_params(t)

    # Half step for momentum
    # negative_potential_x_final, grad_negative_potential_x = value_and_grad(self.negative_potential, argnums=1)(self.params["vd"],x, t)
    negative_potential_x_final, grad_negative_potential_x = self.base_process_value_and_grad(self.params["vd"], x, t)
    xd = xd - eps * grad_negative_potential_x * .5
    negative_potential_xd_final = self.base_process_momentum_potential(xd)
    # Re-sample momemtum
    xd = eta * xd + scale_f * random.normal(rng, xd.shape)
    # Simulate dynamics
    # Half step for momentum, U is the 2nd order potential
    negative_potential_x_init, grad_negative_potential_x = self.base_process_value_and_grad(self.params["vd"], x, t)
    xd_new = xd - eps * .5 * grad_negative_potential_x
    # Full step for x
    negative_potential_xd_init, grad_negative_potential_xd = self.base_process_momentum_value_and_grad(xd)
    x_new = x + eps * grad_negative_potential_xd

    delta_H_init = negative_potential_x_init + negative_potential_xd_init
    # NOTE: I have used different way of calculating delta_H, offset by one half step. Although this can easily be fixed.
    delta_H = delta_H_init - negative_potential_x_final - negative_potential_xd_final

    # TODO: check xd_new is assigned correctly
    # TODO: I have mixed up order to possibly make it compatible with the inner_solver boilerplate
    w = w - self.base_process_momentum_potential(xd_new) + self.base_process_momentum_potential(xd)
    # w = w + log_prob(xd_new, md) - log_prob(xd, md)
    aux = w, delta_H

    # Inner solver goes here
    # Alternate full steps
    # This is an inner solver but it can't be defined within an inner solver update inside a sampler?
    # What is inner step?
    if self.num_leapfrog_steps > 1:
      (rng, x, xd, t) = scan(self.inner_step, (step_rng, x, xd, t), self.inner_ts)[0]

    return x, xd, aux
