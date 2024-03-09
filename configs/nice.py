"""Config for `examples/example.py`."""
from configs.default_config import get_default_configs


def get_config():
    config = get_default_configs()

    # mfvi
    mfvi = config.mfvi
    mfvi.pretrain = True
    mfvi.n_iters = 150  # 150000
    mfvi.lr = 0.01

    # config.N = 5  # 5 for all except NICE
    # training
    training = config.training
    training.n_iters = 150
    training.batch_size = 8

    # data
    data = config.data
    # NICE Config/
    data.im_size = 14
    data.alpha = 0.05
    data.n_bits = 3
    data.hidden_dim = 1000

    # model
    model = config.model
    # for annealed langevin base process
    model.beta_min = 0.01
    model.beta_max = 3.
    model.emd_dim = 48
    # model.num_layers = 3

    # solver
    solver = config.solver
    solver.num_outer_steps = 8
    solver.outer_solver = 'MonteCarloDiffusion'

    return config
