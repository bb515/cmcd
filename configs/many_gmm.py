"""Config for `examples/example.py`."""
from configs.default_config import get_default_configs
import os


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.n_iters = 150  # 150000
    training.batch_size = 8

    # data
    data = config.data
    data.image_size = 2
    data.num_channels = None
    data.num_mixes = 4  # 40
    data.loc_scaling = 10  # 40
    # cwd = os.getcwd()
    # data.file_path = os.path.join(cwd, "../pines.csv")

    # model
    model = config.model
    model.beta_min = 0.01
    model.beta_max = 3.
    model.emd_dim = 48
    # model.num_layers = 3

    # solver
    solver = config.solver
    solver.num_outer_steps = 1000
    solver.outer_solver = 'MonteCarloDiffusion'
    solver.leapfrog_steps = 1
    solver.eps = 0.01
    solver.eta = 0.5
    solver.gamma = 10.0

    return config
