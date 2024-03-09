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
  MCD_SOLVERS,
  UNDERDAMPED_SOLVERS,
  OVERDAMPED_SOLVERS,
)
from cmcd.nn import initialize_mcd_network
import cmcd.opt as opt


def sample_from_target(config, sample_from_target_fn, samples, target_samples, n_samples):
  other_target_samples = sample_from_target_fn(
    jax.random.PRNGKey(2), samples.shape[0]
  )

  calculate_W2_distances(
    samples,
    target_samples,
    other_target_samples,
    n_samples,
    config.training.n_input_dist_seeds,
    n_samples,
  )

  if config.training.use_ema:
    calculate_W2_distances(
      samples_ema,
      target_samples,
      other_target_samples,
      n_samples,
      config.training.n_input_dist_seeds,
      n_samples,
      log_prefix="_ema",
    )


def training(config, log_prob_model, sample_from_target_fn, sample_shape):
  # Set up random seeds
  rng_key_gen = jax.random.PRNGKey(config.seed)
  train_rng_key_gen, eval_rng_key_gen = jax.random.split(rng_key_gen)

  # Train initial variational distribution to maximize the ELBO
  shape = sample_shape
  trainable = ("vd",)
  params_flat, unflatten, params_fixed = initialize(
    config,
    shape,
    trainable=trainable,
    )
  compute_bound_fn = partial(
    compute_bound,
    config=config,
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
      config.mfvi.n_iters,
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
    wandb.log({"elbo_init": jnp.array(elbo_init)})

  if config.solver.outer_solver == "UHA":  # TODO: this is UHA from src/ais_utils.py?
    trainable = ("eta", "mgridref_y")
    if config.training.train_eps:
      trainable = trainable + ("eps",)
    if config.training.train_vi:
      trainable = trainable + ("vd",)
    params_flat, unflatten, params_fixed = initialize(
      config,
      shape=shape,
      vdparams=vdparams_init,
      mdparams=mdparams,
      trainable=trainable
    )

    grad_and_loss = jax.jit(
        jax.grad(compute_bound, 1, has_aux=True), static_argnums=(2, 3, 4), config=config,
    )
    loss = jax.jit(compute_bound, static_argnums=(2, 3, 4))

  elif config.solver.outer_solver in MCD_SOLVERS:
    trainable = ("eta", "gamma", "mgridref_y")
    if config.training.train_eps:
      trainable = trainable + ("eps",)
    if config.training.train_vi:
      trainable = trainable + ("vd",)

    print(f"Params being trained : {trainable}")
    params_flat, unflatten, params_fixed = initialize(
      config=config,
      shape=shape,
      vdparams=vdparams_init,
      trainable=trainable,
    )

    if "var" in config.solver.outer_solver.lower():
      compute_bound_fn = partial(
        compute_bound_var,
        config=config,
      )
    else:
      compute_bound_fn = partial(
        compute_bound,
        config=config,
      )

    grad_and_loss = jax.jit(
      jax.grad(compute_bound_fn, 1, has_aux=True), static_argnums=(2, 3, 4),
    )
    loss_fn = jax.jit(compute_bound_fn, static_argnums=(2, 3, 4))

  else:
    raise NotImplementedError("Mode %s not implemented." % config.solver.bound_mode)

  # Average over 30 seeds, 500 samples each after training is done.
  n_samples = config.training.n_samples
  n_input_dist_seeds = config.training.n_input_dist_seeds

  if sample_from_target_fn is not None:
    target_samples = sample_from_target_fn(
      jax.random.PRNGKey(1), n_samples * n_input_dist_seeds
    )
  else:
    target_samples = None

  _, params_flat, ema_params = opt.run(
    config,
    config.training.lr,
    config.training.n_iters,
    params_flat,
    unflatten,
    params_fixed,
    log_prob_model,
    grad_and_loss,
    trainable,
    train_rng_key_gen,
    log_prefix="train",
    target_samples=target_samples,
    use_ema=config.training.use_ema,
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

  if config.training.use_ema:
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

  params_train, params_notrain = unflatten(params_flat)
  params = {**params_train, **params_notrain}
  return params, samples, target_samples, n_samples


def initialize(
    config,
    shape,
    trainable=["eps", "eta"],
    vdparams=None,
    mdparams=None):
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
  for param in ["vd", "eps", "eta", "md", "gamma"]:
    if param=="vd":  # variational? distribution
      init_param = vdparams
      if vdparams is None:
        init_param = initialize_dist(shape, init_sigma=config.solver.init_sigma)
    elif param=="md":  # momentum? distribution
      init_param = mdparams
      if mdparams is None:
        init_param = jnp.zeros(shape)
    elif param=="eta":
      init_param = config.solver.eta
    elif param=="eps":
      init_param = config.solver.eps
    elif param=="gamma":
      init_param = config.solver.gamma
    if param in trainable:
      params_train[param] = init_param
    else:
      params_notrain[param] = init_param

  if config.solver.outer_solver in OVERDAMPED_SOLVERS:
  # if solver in [  # Overdamped methods
  #     # TODO: what solvers do these correspond to?
  #   "ULA_sn", "UHA?"
  #   "U_elpsna","LeapfrogEA"?
  #   "U_alpsna", "LeapfrogA"?
  #   "CMCD_sn",  "CMCDOD"?
  #   "CMCD_var_sn", "CMCDODVar"?
  # ]:
    # TODO: intialize_mcd_network
    x_dim = shape[0]
    in_dim = x_dim + emb_dim
    init_sn, apply_sn = initialize_mcd_network(x_dim, in_dim, emb_dim, num_outer_steps)
    params_train["sn"] = init_sn(rng, None)[1]
  elif config.solver.outer_solver in UNDERDAMPED_SOLVERS:
  # elif solver in [  # Underdamped methods
  #     "U_alpsn",  "LeapfrogA"
  #     "U_ealpsn",  "LeapfrogEA"
  #     "U_anvsn",  "?"
  #     "CAIS_UHA_sn",  "LeapfrogACAIS"
  #   ]:
    x_dim = shape[0]
    xd_dim = shape[0]
    in_dim = x_dim + xd_dim + emb_dim
    init_sn, apply_sn = initialize_mcd_network(x_dim, in_dim, emb_dim, num_outer_steps)
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
  params_fixed = (shape, num_outer_steps, apply_sn)
  params_flat, unflatten = ravel_pytree((params_train, params_notrain))
  return params_flat, unflatten, params_fixed


def compute_ratio(seed, params_flat, unflatten, params_fixed, log_prob, config):
  rng = jax.random.PRNGKey(seed)

  # If params not supplied here:
  # params_flat, params_fixed = initialize_bm()
  params_train, params_notrain = unflatten(params_flat)
  params_notrain = jax.lax.stop_gradient(params_notrain)
  params = {**params_train, **params_notrain}  # All parameters in a single place
  shape, num_steps, apply_sn = params_fixed
  assert num_steps == config.solver.num_outer_steps

  # TODO: Need to reinitiate solver for every score, but this should be fixed and factored out of training loop?
  target_x, gridref_x, mgridref_y, ts = get_betas(num_steps)
  beta, _ = get_linear_beta_function(beta_min=config.model.beta_min, beta_max=config.model.beta_max)

  # TODO: may need flag for mcbm vs bm methods here since I think mcbm methods don't have auxilliary score?
  # TODO: boundingmachine method returns delta_H whereas mcbounding_machine does not return delta_H
  # auxilliary_process_score = get_score(model, params, score_scaling)

  base_process_potential = get_annealed_langevin(log_prob)
  # TODO: is it okay to parameterize parameters here?
  # base_process_score = jax.grad(lambda x, t: base_process_potential(params["vd"]))
  base_process_score = jax.grad(base_process_potential, argnums=1)
  # (params, log_prob, base_process_score=None, auxilliary_process_score=None, beta=None, ts=None, gamma=10., clip=1e2)
  # TODO: this not working for MonteCarloDiffusion Solver as arguments are
  # TODO: reconsile betas and ts
  outer_solver = get_solver(config, params, log_prob, base_process_score, apply_sn, beta, ts)

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
  x, aux, num_function_evaluations = sampler(sample_rng)

  if config.solver.outer_solver.lower() == "uha":
    w, delta_H = aux
  else:
    w = aux

  return -1. * w, x
  # TODO: not sure when aux is a tuple... fix?
  # TODO: lax.pmean needed?
  # delta_H = jnp.mean(jnp.abs(delta_H))
  # return -1. * w, (x, delta_H)


# @functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob, config):
  # # TODO: decided on whether to explicitly use a batch dimension and take away the vmap,
  # # at the moment the only reason to keep explicit batch dimension is because Song's models
  # # explicitly batch across this dimension without vmap
  ratios, x = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None, None))(
    seeds, params_flat, unflatten, params_fixed, log_prob, config,
  )
  return ratios.mean(), (ratios, x)


def compute_bound_var(seeds, params_flat, unflatten, params_fixed, log_prob, config):
  ratios, x = jax.vmap(compute_ratio, in_axes=(0, None, None, None, None, None))(
    seeds, params_flat, unflatten, params_fixed, log_prob, config
  )
  return jnp.clip(ratios.var(ddof=0), -1e7, 1e7), (ratios, x)


def get_solver(config, params, log_prob, base_process_score, apply_sn, beta, ts):
  if config.solver.outer_solver.lower()=="cmcdud":
    Solver = CMCDUD
  elif config.solver.outer_solver.lower()=="leapfrogea":
    Solver = LeapfrogEA
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
    Solver = MonteCarloDiffusion
  elif config.solver.outer_solver.lower()=="uha":
    return UHA(params, log_prob, base_process_score, apply_sn, beta, ts, num_leapfrog_steps=config.solver.num_leapfrog_steps)
  else:
    raise NotImplementedError(f"Solver {config.solver.outer_solver} unknown.")
  return Solver(params, log_prob, base_process_score, apply_sn, beta, ts)


def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
  config_dict.unlock()
  config_dict.update_from_flattened_dict(run.config)
  config_dict.update_from_flattened_dict(new_vals)
  run.config.update(new_vals, allow_val_change=True)
  config_dict.lock()


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
