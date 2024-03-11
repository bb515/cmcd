import ml_collections
import os


def get_default_configs():
  config = ml_collections.ConfigDict()

  config.mfvi = mfvi = ml_collections.ConfigDict()
  mfvi.pretrain = True
  mfvi.n_iters = 150  # 150000
  mfvi.lr = 0.01

  config.model = model = ml_collections.ConfigDict()
  # for annealed langevin
  model.eps_schedule = 'cos_sq'
  model.beta_min = 0.1
  model.beta_max = 25.0

  # training
  config.training = training = ml_collections.ConfigDict()
  training.n_iters = 150  # 150000
  # TODO: investigate if an SDE can be used to define the base process
  # training.sde = 'vpsde'
  # training.sde = 'vesde'
  # TODO: implement typical score network training parameters
  # training.likelihood_weighting = False
  # training.score_scaling = True
  # training.reduce_mean = True
  training.batch_size = 5  # 5 for all except NICE
  # TODO: implement jit compiled training steps of solver
  # training.n_jitted_steps = 1
  # TODO: implement pmap over devices in training
  # training.pmap = False
  # TODO: implement training monitoring
  # training.log_epoch_freq = 1
  # training.log_step_freq = 8000
  # TODO: implement store additional checkpoints for preemption in cloud computing environments
  # training.snapshot_freq = 8000
  # training.snapshot_freq_for_preemption = 8000
  # training.eval_freq = 8000
  training.run_cluster = False
  training.train_eps = True
  training.train_vi = True
  training.train_betas = True
  training.n_samples = 500
  training.n_input_dist_seeds = 30
  training.lr = 0.0001
  training.use_ema = False
  training.grad_clip = False
  training.traj_bal = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.stack_samples = False
  sampling.denoise = True

  # TODO: implement evaluation
  # # evaluation
  # config.eval = evaluate = ml_collections.ConfigDict()
  # evaluate.batch_size = 128

  # data
  config.data = data = ml_collections.ConfigDict()

  # model
  config.model = model = ml_collections.ConfigDict()
  # TODO: implement diffusionjax model code
  # model.name = 'mlp'
  model.emb_dim = 2
  # model.nlayers = 3

  # solver
  config.solver = solver = ml_collections.ConfigDict()
  solver.num_outer_steps = 8
  solver.outer_solver = 'MonteCarloDiffusion'
  solver.sigma = 1.0  # TODO: should be model?
  solver.num_leapfrog_steps = 1
  solver.eps = 0.01
  solver.eta = 0.5  # TODO: fix name conflict, this is a discrete_beta
  solver.gamma = 10.0

  # optimization
  config.seed = 2023
  config.optim = optim = ml_collections.ConfigDict()
  # TODO: implement diffusionjax optimizer code
  # optim.optimizer = 'Adam'
  # optim.lr = 2e-4
  # optim.warmup = 5000
  # optim.weight_decay = False
  # optim.grad_clip = None
  # optim.beta1 = 0.9
  # optim.eps = 1e-8

  # TODO: implement conditional sampling
  # # sampling
  # sampling = config.sampling
  # sampling.cs_method = None
  # sampling.noise_std = 1.0
  # sampling.denoise = True  # work out what denoise_override is
  # sampling.innovation = True  # this will probably be superceded
  # sampling.inverse_scaler = None
  # sampling.stack_samples = False

  # Wandb Configs
  config.wandb = ml_collections.ConfigDict()
  config.wandb.log = False
  config.wandb.project = "cmcd"
  config.wandb.entity = "bb515"
  cwd = os.getcwd()
  config.wandb.code_dir = cwd
  config.wandb.name = ""
  config.wandb.log_artifact = True

  return config
