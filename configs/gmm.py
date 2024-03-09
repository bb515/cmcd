"""Config for `gmm.py`."""
from configs.default_config import get_default_configs


def get_config():
    config = get_default_configs()

    # mfvi
    mfvi = config.mfvi
    mfvi.pretrain = True
    mfvi.n_iters = 150  # 150000
    mfvi.lr = 0.01

    # training
    training = config.training
    training.n_iters = 150  # 150000
    training.batch_size = 8

    # data
    data = config.data

    # model
    model = config.model
    model.emb_dim = 20
    model.beta_min = 0.1
    model.beta_max = 25.0

    # solver
    solver = config.solver
    solver.num_outer_steps = 8
    solver.outer_solver = 'MonteCarloDiffusion'

    return config
