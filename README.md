# Transport meets Variational Inference: Controlled Monte Carlo Diffusions

(A non official) implementation for the ICLR 2024 paper [[OpenReview]](https://openreview.net/forum?id=PP1rudnxiW) [[arXiv]](https://arxiv.org/abs/2307.01050).

We provide code to run the experiments in the paper, on a wide variety of target distributions that have been implemented in `cmcd/examples/`. The code is written in Jax, and we use `wandb` for logging and visualisation.

To run different methods and targets, following the template below - 

```python main.py --config.model log_ionosphere --config.solver.outer_solver CMCDOD```

For the `config.solver.outer_solver`
- CMCD, ULA and MCD use `CMCDOD`
- UHA uses `UHA`
- LDVI uses `LeapfrogA`
- 2nd order CMCD uses `LeapfrogACAIS`
- CMCD + VarGrad loss uses `VarCMCDOD`

Contents:
- [Installation](#installation)
- [Experiments](#experiments)
    - [Gaussian Mixture Model, 40 modes](#40-gmm)
    - [Funnel](#funnel)
    - [LGCP](#lgcp)
    - [Gaussian Mixture Model, 2 modes](#2-gmm)

## Installation
The package requires Python 3.9. First, it is recommended to [create a new python virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

Then, install `jax`. Note the JAX installation is different depending on your CUDA version.

Make sure the jax versions are working on your accelerator. Then,
- Install `diffusionjax`.
- Clone the repository `git clone git@github.com:bb515/cmcd.git`.
- Install using pip `pip install -e .` from the working directory of this README file (see the `setup.py` for the requirements that this command installs).

## Experiments

TODO: Complete this section to validate implementation. Below, we provide the commands replicating the hparam settings used in the paper, and the wandb links to the experiments.

#### 40-GMM

By default, in order to make comparisons to DDS/PIS, we use the same network architecture with time embeddings from the DDS repo. In order to run our method using the DDS architecture, you can set `--config.nn_arch dds` in the command line.

```bash
python examples/many_gmm.py --config.solver.outer_solver CMCDOD --config.N 2000 --config.num_outer_steps 256 --noconfig.mfvi.pretrain --config.solver.sigma 60 --config.training.grad_clip --config.solver.eps 1 --config.model.eps_schedule cos_sq --config.training.lr 0.001 --noconfig.training.train_eps --noconfig.training.train_vi --config.wandb.name "kl 40gmm pis net eps=1, cos_sq" --config.nn_arch dds
```

```bash
-python examples/many_gmm.py -config.solver.outer_solver MCD_CAIS_var_sn --config.N 2000 --config.num_outer_steps 256 --noconfig.mfvi.pretrain --config.solver.sigma 15 --config.training.grad_clip --config.solver.eps 0.65 --config.model.emb_dim 130 --config.training.lr 0.005 --noconfig.training.train_eps --noconfig.training.train_vi --config.wandb.name "logvar 40gmm"
```

```bash
python examples/many_gmm.py --config.solver.outer_solver CMCDOD --config.N 2000 --config.num_outer_steps 256 --noconfig.mfvi.pretrain --config.solver.sigma 15 --config.training.grad_clip --config.solver.eps 0.1 --config.model.emb_dim 130 --config.training.lr 0.005 --noconfig.training.train_eps --noconfig.training.train_vi --config.wandb.name "kl 40gmm"
```

```bash
 python examples/many_gmm.py --config.solver.outer_solver CMCDOD --config.N 2000 --config.num_outer_steps 256 --noconfig.mfvi.pretrain --config.solver.sigma 60 --config.training.grad_clip --config.solver.eps 1 --config.model.eps_schedule cos_sq --config.training.lr 0.001 --noconfig.training.train_eps --noconfig.training.train_vi --config.wandb.name "kl 40gmm pis net eps=1, cos_sq" --config.nn_arch dds
```

#### Funnel

```bash
python examples/funnel.py --config.solver.outer_solver CMCDOD --config.N 300 --config.data.alpha 0.05 --config.model.emb_dim 48 --config.solver.eps 0.1 -config.solver.sigma 1 --config.training.n_iters 11000 --noconfig.mfvi.pretrain --config.training.train_vi --noconfig.training.train_eps --config.wandb.name "funnel replicate w/ cos_sq" --config.training.lr 0.01 --config.training.n_samples 2000 --config.model.eps_schedule cos_sq
```

#### LGCP

```
python examples/lgcp.py --config.solver.outer_solver CMCDOD --config.N 20 --config.data.alpha 0.05 --config.model.emb_dim 20 --config.solver.eps 0.00001 -config.solver.sigma 1 --config.training.n_iters 37500 --config.mfvi.pretrain --config.training.train_vi --config.training.train_eps --config.wandb.name "lgcp replicate" --config.training.lr 0.0001 --config.training.n_samples 500 --config.mfvi.iters 20000
```

#### 2-GMM

```
python examples/gmm.py --config.solver.outer_solver CMCDOD --config.N 300 --config.data.alpha 0.05 --config.model.emb_dim 20 --config.solver.eps 0.01 -config.solver.sigma 1 --config.training.n_iters 11000 --noconfig.mfvi.pretrain --config.training.train_vi --noconfig.training.train_eps --config.wandb.name "gmm replicate" --config.training.lr 0.001 --config.training.n_samples 500
```

If you use any of this code, please cite the original work using the following BibTeX entry:

```bibtex
@inproceedings{
vargas2024transport,
title={Transport meets Variational Inference: Controlled Monte Carlo Diffusions},
author={Francisco Vargas and Shreyas Padhy and Denis Blessing and Nikolas N{\"u}sken},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=PP1rudnxiW}
}
```
