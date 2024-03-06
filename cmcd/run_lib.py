import os
import pickle
from functools import partial
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import ml_collections.config_flags
import numpy as np
import wandb
from absl import app, flags
from jax.config import config as jax_config
from diffusionjax.utils import(
  flatten_nested_dict,
  get_linear_beta_function,
  continuous_to_discrete,
  get_times,
  get_timestep,
)
from cmcd.utils import (
  get_underdamped_sampler,
  get_overdamped_sampler,
  get_annealed_langevin,
  calculate_W2_distances,
  log_final_losses,
  make_grid,
  initialize_dist,
  get_betas,
  )
from cmcd.solvers import (
  CMCDUD,
  LeapfrogEA,
  LeapfrogA,
  LeapfrogE,
  LeapfrogME,
  LeapfrogACAIS,
  CMCDOD,
  VarCMCDOD,
  MonteCarloDiffusion,
  UHA,
  UNDERDAMPED_SOLVERS,
  OVERDAMPED_SOLVERS,
)


def initialize(
    config,
    shape,
    vdparams=None,
    mdparams=None,
    trainable=["eps", "eta"],
    init_sigma=1.,
    solver=None):
  """
  Solvers allowed:
    - ULA: This is ULA. Method from Thin et al.
    - ULA_sn: This is MCD. Method from Doucet et al.
    - U_a-lp: UHA but with approximate sampling of momentum (no score network).
    - U_a-lp-sn: Approximate sampling of momentum, followed by leapfrog, using score network(x, rho) for backward sampling.
    - CAIS_sn: CAIS with trainable SN.
    - CAIS_UHA_sn: CAIS underdampened with trainable SN.
  """
  emb_dim = config.model.emb_dim  # TODO check
  rng = jax.random.PRNGKey(config.seed)
  num_outer_steps = config.solver.num_outer_steps

  params_train = {}  # all trainable parameters
  params_notrain = {}  # all non-trainable parameters
  for param in ["vd", "eps", "eta", "md"]:
    if param in trainable:
      if param=="vd":
        init_param = initialize_dist(dim, init_sigma=init_sigma)
      elif param=="md":
        init_param = md.initialize(dim)
      elif param=="eta":
        init_param = config.solver.eta
      elif param=="eps":
        init_param = config.solver.eta

      params_train[param] = init_param
    else:
      params_notrain[param] = init_param

  if solver in [  # Overdamped methods
      # TODO: what solvers do these correspond to?
    "ULA_sn",
    "U_elpsna",
    "U_alpsna",
    "CMCD_sn",
    "CMCD_var_sn",
  ]:
    # TODO: intialize_mcd_network
    init_sn, apply_sn = initialize_mcd_network(dim, emb_dim, num_outer_steps, num_layers=config.model.num_layers)
    params_train["sn"] = init_sn(rng, None)[1]
  elif solver in [  # Underdamped methods
      "U_alpsn",
      "U_ealpsn",
      "U_anvsn",
      "CAIS_UHA_sn",
    ]:
    init_sn, apply_sn = initialize_mcd_network(dim, emb_dim, num_outer_steps, xd_dim=xd_dim, num_layers=config.model.num_layers)
    params_train["sn"] = init_sn(rng, None)[1]
  else:
    # TODO: No score network in use? since not mcbm but bm?
    # TODO: switchmode for score network which does not apply to boundingmachine methods
    # it may apply to mcboundingmachinemethods
    apply_sn = None
    print("No score network needed by the method.")

  # Everything related to betas
  target_x, gridref_x, mgridref_y, ts = get_betas(num_outer_steps)
  params_notrain["gridref_x"] = gridref_x
  # BB: Does it not make sense to start this at 1. since density must be evaluated at 1. where the prior sample is initiated.
  # at the distribution that one starts in? i.e., shouldn't it instead be: # params_notrain["target_x"] = jnp.linspace(0, 1, num_outer_steps + 1)[0:-1]
  params_notrain["target_x"] = target_x

  if "mgridref_y" in trainable:
    params_train["mgridref_y"] = mgridref_y
  else:
    params_notrain["mgridref_y"] = mgridref_y

  # Other fixed parameters
  params_fixed = (dim, num_outer_steps, apply_sn)
  params_flat, unflatten = ravel_pytree((params_train, params_notrain))
  return params_flat, unflatten, params_fixed


def compute_ratio(rng, params_flat, unflatten, params_fixed, log_prob, config, shape):

  # If params not supplied here:
  # params_flat, params_fixed = initialize_bm()
  params_train, params_notrain = unflatten(params_flat)
  params_notrain = jax.lax.stop_gradient(params_notrain)
  params = {**params_train, **params_notrain}  # All parameters in a single place
  dim, num_steps, apply_sn = params_fixed
  assert num_steps == config.solver.num_outer_steps

  # TODO: Need to reinitiate solver for every score, but this should be fixed and factored out of training loop?
  target_x, gridref_x, mgridref_y, ts = get_betas(num_steps)
  beta, _ = get_linear_beta_function(beta_min=config.model.beta_min, beta_max=config.model.beta_max)

  # TODO: may need flag for mcbm vs bm methods here since I think mcbm methods don't have auxilliary score?
  # TODO: boundingmachine method returns delta_H whereas mcbounding_machine does not return delta_H
  # auxilliary_process_score = get_score(model, params, score_scaling)

  base_process_potential = get_annealed_langevin(log_prob)
  base_process_score = jax.grad(base_process_potential)
  # (params, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None, gamma=10., clip=1e2)
  # TODO: this not working for MonteCarloDiffusion Solver as arguments are
  # TODO: reconsile betas and ts
  outer_solver = get_solver(config, params, shape, log_prob, base_process_score, apply_sn, beta, ts)

  # Evolve sampler and update weight
  # TODO: define inverse scaler
  inverse_scaler = None
  if config.solver.outer_solver in UNDERDAMPED_SOLVERS:
     get_sampler = get_underdamped_sampler
  elif config.solver.outer_solver in OVERDAMPED_SOLVERS:
    get_sampler = get_overdamped_sampler
  else:
    raise NotImplementedError("Not implemented sampler for the solver")

  sampler = get_sampler(
    shape, outer_solver, denoise=config.sampling.denoise,
    stack_samples=False, inverse_scaler=inverse_scaler)

  rng, sample_rng = jax.random.split(rng, 2)
  # NOTE: The only thing that needs batching is this sampler, so I don't think that compute_ratio needs to be vmapped
  q_samples, aux, num_function_evaluations = sampler(sample_rng)
  # TODO: not sure aux is a tuple... fix?
  w, delta_H = aux
  delta_H = jnp.max(jnp.abs(delta_H))
  # TODO: lax.pmean needed?
  # delta_H = jnp.mean(jnp.abs(delta_H))
  return -1. * w, (x, delta_H)


# @functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(rng, params_flat, unflatten, params_fixed, log_prob, config, shape):
  # NOTE: I decided to take away the vmap, and explicitly batch across a batch axis of dim batch_size
  ratios, (z, _) = compute_ratio(
    rng, params_flat, unflatten, params_fixed, log_prob, config, shape,
  )
  # ratios, (z, _) = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None, None, None))(
  #   seeds, params_flat, unflatten, params_fixed, log_prob, config, shape,
  # )
  print(ratios.shape)
  assert 0
  return ratios.mean(), (ratios, z)


def compute_bound_var(rng, params_flat, unflatten, params_fixed, log_prob, config, shape):
  ratios, (z, _) = compute_ratio(
    rng, params_flat, unflatten, params_fixed, log_prob, config, shape,
  )
  # ratios, (z, _) = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None, None, None))(
  #   seeds, params_flat, unflatten, params_fixed, log_prob, config
  # )
  return jnp.clip(ratios.var(ddof=0), -1e7, 1e7), (ratios, z)


def get_solver(config, params, shape, log_prob, base_process_score, apply_sn, beta, ts):
  if config.solver.outer_solver.lower()=="cmcdud":
    Solver = CMCDUD
  elif config.solver.outer_solver.lower()=="leapfrogea":
    Solver = LeapFrogEA
  elif config.solver.outer_solver.lower()=="leapfroga":
    Solver = LeapfrogA
  elif config.solver.outer_solver.lower()=="leapfroge":
    Solver = LeapfrogE
  elif config.solver.outer_solver.lower()=="leapfrogme":
    Solver = LeapfrogME
  elif config.solver.outer_solver.lower()=="leapfrogacais":
    Solver = LeapfrogACAIS
  elif config.solver.outer_solver.lower()=="cmcdod":
    Solver = CMCDOD
  elif config.solver.outer_solver.lower()=="varcmcdod":
    Solver = VarCMCDOD
  elif config.solver.outer_solver.lower()=="montecarlodiffusion":
    return MonteCarloDiffusion(log_prob, base_process_score, apply_sn, beta, ts)
  elif config.solver.outer_solver.lower()=="uha":
    return UHA(params, shape, log_prob, base_process_score, apply_sn, beta, ts, num_leapfrog_steps=config.solver.num_leapfrog_steps)
  else:
    raise NotImplementedError(f"Solver {config.solver.outer_solver} unknown.")
  return Solver(params, shape, log_prob, base_process_score, apply_sn, beta, ts)


def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
  config_dict.unlock()
  config_dict.update_from_flattened_dict(run.config)
  config_dict.update_from_flattened_dict(new_vals)
  run.config.update(new_vals, allow_val_change=True)
  config_dict.lock()


def setup_config(wandb_config, config):
  try:
    if wandb_config.model == "nice":
      config.model = (
        wandb_config.model
        + "_{}_{}_{}".format(wandb_config.alpha, wandb_config.n_bits, wandb_config.im_size))
      new_vals = {}
    elif wandb_config.model in ["funnel"]:
      pass
  except KeyError:
    new_vals = {}
    print("LR not found for model {} and boundmode {}".format(
      wand_config.model, wandb_config.boundmode))
  return new_vals


def setup_training(wandb_run):
  """Helper function that sets up training configs and logs to wandb."""
  if not wandb_run.config.get("use_tpu", False):
    # TF can hog GPU memory, so hide the GPU device from it
    # tf.config.experimental.set_visible_devices([], "GPU")

    # Without this, JAX is automatically using 90% GPU for pre-allocation
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
    # os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    # Disable logging of compiles
    jax.config.update("jax_log_compiles", False)

    # Log various JAX configs to wandb, and locally
    wandb_run.summary.update(
      {
        "jax_process_index": jax.process_index(),
        "jax.process_count": jax.process_count(),
      }
    )
  else:
    # config.FLAGS.jax_xla_backend = "tpu_driver"
    # config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
    # DEVICE_COUNT = len(jax.local_devices())
    print(jax.default_backend())
    print(jax.device_count(), jax.local_device_count())
    print("8 cores of TPU (Local devices in JAX):")
    print("\n".join(map(str, jax.local_devices())))
