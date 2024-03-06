"""Config for `gmm.py`."""
from configs.default_config import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vpsde'
    # training.sde = 'vesde'
    training.n_iters = 4000
    training.batch_size = 8
    training.likelihood_weighting = False
    training.score_scaling = True
    training.reduce_mean = True
    training.log_epoch_freq = 1
    training.log_step_freq = 8000
    training.pmap = False
    training.n_jitted_steps = 1
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq = 8000
    training.snapshot_freq_for_preemption = 8000
    training.eval_freq = 8000

    # eval
    eval = config.eval
    eval.batch_size = 1000

    # sampling
    sampling = config.sampling
    sampling.cs_method = None
    sampling.noise_std = 1.0
    sampling.denoise = True  # work out what denoise_override is
    sampling.innovation = True  # this will probably be superceded
    sampling.inverse_scaler = None
    sampling.stack_samples = False

    # data
    data = config.data
    data.image_size = 2
    data.num_channels = None

    # model
    model = config.model
    # for vp
    model.beta_min = 0.1
    model.beta_max = 25.0  # 200 also works, depends on time step size
    # for ve
    model.sigma_min = 0.01
    model.sigma_max = 10.

    model.emb_dim = 20
    model.num_layers = 3

    # solver
    solver = config.solver
    solver.num_outer_steps = 8
    solver.eta = 1.0
    solver.inner_solver = None
    solver.stsl_scale_hyperparameter = 0.02
    solver.dps_scale_hyperparameter = 0.05
    solver.init_sigma = 1.0
    # solver.outer_solver = 'UHA'
    solver.outer_solver = 'CMCDUD'
    solver.num_leapfrog_steps = 1

    # mfvi
    mfvi = config.mfvi
    mfvi.pretrain = True
    mfvi.iters = 150000
    mfvi.lr = 0.01

    # optim
    optim = config.optim
    optim.optimizer = 'Adam'
    optim.lr = 1e-3
    optim.warmup = False
    optim.weight_decay = False
    optim.grad_clip = None

    config.seed = 2023

    return config
