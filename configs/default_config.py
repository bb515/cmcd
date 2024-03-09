import ml_collections
import os


def get_default_configs():
  config = ml_collections.ConfigDict()

  # what is mfvi?
  config.mfvi = mfvi = ml_collections.ConfigDict()
  mfvi.pretrain = True
  mfvi.n_iters = 150  # 150000
  mfvi.lr = 0.01

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 5
  training.n_iters = 150  # 150000
  training.snapshot_freq = 50000
  training.log_epochs_freq = 10
  training.log_step_freq = 8
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.score_scaling = True
  training.n_jitted_steps = 1
  training.pmap = False
  training.reduce_mean = True
  training.pointwise_t = False

  training.run_cluster = False

  training.train_eps = True
  training.train_vi = True
  training.n_samples = 500
  training.n_input_dist_seeds = 30
  training.lr = 0.0001

  training.use_ema = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.stack_samples = False
  sampling.denoise = True

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 128

  # data
  config.data = data = ml_collections.ConfigDict()
  data.num_channels = None
  data.image_size = 2

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'mlp'

  # for vp
  model.beta_min = 0.1
  model.beta_max = 20.

  # for ve
  model.sigma_max = 378.
  model.sigma_min = 0.01

  # cmcd model stuff for the default network that they used
  model.emb_dim = 2
  model.nlayers = 3

  # solver
  config.solver = solver = ml_collections.ConfigDict()
  solver.num_outer_steps = 8
  solver.num_inner_steps = 1
  solver.outer_solver = 'EulerMaruyama'
  solver.eta = None  # for DDIM
  solver.inner_solver = None
  solver.dt = None
  solver.epsilon = None
  solver.snr = None
  solver.init_sigma = 1.0
  solver.bound_mode = 'UHA'
  solver.num_leapfrog_steps = 1

  # optimization
  config.seed = 2023
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.warmup = 5000
  optim.weight_decay = False
  optim.grad_clip = None
  optim.beta1 = 0.9
  optim.eps = 1e-8

  # Wandb Configs
  config.wandb = ml_collections.ConfigDict()
  config.wandb.log = True
  config.wandb.project = "cmcd"
  config.wandb.entity = "bb515"
  cwd = os.getcwd()
  config.wandb.code_dir = cwd
  config.wandb.name = ""
  config.wandb.log_artifact = True

  return config
