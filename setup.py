"""
Setup script for cmcd.

This setup is required or else
    >> ModuleNotFoundError: No module named 'cmcd'
will occur.
"""
from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

extra_compile_args = ['-O3']
extra_link_args = []


setup(
    name="cmcd",
    version="0.0.0",
    description="cmcd is a package for solving sampling problems",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=['*.test']),
    install_requires=[
    #     'diffusionjax',
    #     'flax==0.3.3',
      'ml-collections==0.1.1',
      'numpy',
      'scipy',
      'h5py',
      'matplotlib',
      'absl-py==0.10.0',
      'opt-einsum==3.3.0',
      'optax==0.1.3',
      'pickleshare==0.7.5',
      'tqdm==4.64.1',
      'wandb==0.13.5',
    ],
    extras_require={
        'linting': [
          "flake8",
          "pylint",
          "mypy",
          "typing-extensions",
          "pre-commit",
          "ruff",
          'jaxtyping',
        ],
        'testing': [
          "pytest",
          "pytest-xdist",
          "pytest-cov",
          "coveralls",
          "jax>=0.4.1",
          "jaxlib>=0.4.1",
          "setuptools_scm[toml]",
          "setuptools_scm_git_archive",
        ],
        'examples': [
          "chex==0.1.82",
          "inference_gym==0.0.4",
          "haiku",
          "dm-haiku==0.0.9",
          "dm-tree==0.1.7",
          "distrax==0.1.2",
          "keras==2.13.1",
          "matplotlib==3.6.2",
          "matplotlib-inline==0.1.6",
          "numpyro==0.10.1",
        ],
    },
    include_package_data=True)
