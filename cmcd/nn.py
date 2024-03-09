import jax
import jax.numpy as jnp
from jax.example_libraries.stax import (
    Dense,
    FanInSum,
    FanOut,
    Identity,
    Softplus,
    parallel,
    serial,
)
# TODO: convert to Flax


def initialize_embedding(rng, num_steps, emb_dim, factor=0.05):
  return jax.random.normal(rng, shape=(num_steps, emb_dim)) * factor


def initialize_mcd_network(x_dim, in_dim, emb_dim, num_steps):
  layers = [
      serial(
          FanOut(2), parallel(Identity, serial(Dense(in_dim), Softplus)), FanInSum
      ),
      serial(
          FanOut(2), parallel(Identity, serial(Dense(in_dim), Softplus)), FanInSum
      ),
      Dense(x_dim),
  ]
  init_fun_nn, apply_fun_nn = serial(*layers)

  def init_fun(rng, input_shape):
    params = {}
    output_shape, params_nn = init_fun_nn(rng, (in_dim,))
    params["nn"] = params_nn
    rng, _ = jax.random.split(rng)
    params["emb"] = initialize_embedding(rng, num_steps, emb_dim)
    params["factor_sn"] = jnp.array(0.0)
    return output_shape, params

  def apply_fun(params, inputs, i, **kwargs):
    # inputs has size (x_dim)
    print(params["emb"])
    emb = params["emb"][i, :]  # (emb_dim,)
    input_all = jnp.concatenate([inputs, emb])
    return apply_fun_nn(params["nn"], input_all) * params["factor_sn"]  # (x_dim,)

  return init_fun, apply_fun

  # def apply_fun(params, inputs, i, **kwargs):
  #   # inputs has size (x_dim)
  #   emb = params["emb"][i, :]  # (emb_dim,)
  #   input_all = jnp.concatenate([inputs, emb])
  #   return apply_fun_nn(params["nn"], input_all) * params["factor_sn"]  # (x_dim,)
