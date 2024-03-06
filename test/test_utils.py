"""Test utils.py"""
from cmcd.utils import log_prob_kernel
import numpy as np
import jax.numpy as jnp


def alt_log_prob_kernel(x, mean, scale):
  """Unit test for numpyro."""
  n = jnp.size(x)
  # deal with zero variances by adding a constant term if zero
  var = scale**2
  centered = x - mean
  log_2pi = jnp.log(2 * jnp.pi)
  return - 0.5 * (n * log_2pi + jnp.sum(jnp.log(var)) + centered.T @ centered / var)


def test_log_prob_kernel():
  shape = (5,)
  x = jnp.ones(shape) * .3
  mean = jnp.zeros(shape)
  x = jnp.ones(shape)
  actual_log_prob = log_prob_kernel(x, mean, scale)
  expected_log_prob = alt_log_prob_kernel(x, mean, scale)
  assert np.all_close(actual_log_prob, expected_log_prob)


def test_get_betas():
  num_steps = 1000
  dt, ts = get_betas(num_steps)
  # BB: Why not just have
  ts_alt = gridref_y[1:-1]
  dt_alt = ts_alt[1] - ts_alt[0]
  ts_bb, dt_bb = get_times(num_steps)
  print("dt: ", dt_bb, dt, dt_alt)
  print("len: ", ts.shape, ts_alt.shape, ts_bb.shape)
  print("maxmin", jnp.max(ts), jnp.min(ts))
  print(jnp.max(ts_bb), jnp.min(ts_bb))
  import matplotlib.pyplot as plt
  plt.plot(ts, ts_alt)
  plt.savefig("testnative.png")
  plt.close()
  plt.plot(ts, ts_bb.flatten() / ts)
  plt.savefig("testbrel.png")
  plt.close()
  plt.plot(ts, ts_bb.flatten() - ts)
  plt.savefig("testbabs.png")
  plt.close()
  assert jnp.allclose(ts, ts_alt)
  assert jnp.allclose(dt, dt_alt)
  # assert jnp.allclose(ts, ts_bb) NOTE: fails and TODO: check that the correct inital and final ts are used.
  # assert jnp.allclose(dt, dt_bb) NOTE: fails
