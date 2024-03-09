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

  # TODO: delete
  # get_betas(num_steps=1000)

  # from jax import vmap
  # num_steps = 1000
  # t1 = 1.0
  # t0 = 0.001
  # dt = 0.001
  # offset = 0.8
  # ii = jnp.linspace(0, 999, num_steps, dtype=int)
  # print(ii)
  # print(dt)

  # SSlinear = vmap(lambda i: SSlinear_beta_schedule(num_steps, t1, i, final_eps=t0))(ii)
  # SScosine = vmap(lambda i: SScosine_beta_schedule(num_steps, t1, i, s=offset))(ii)

  # linear, _dt, _t0, _t1 = get_linear_beta_schedule(num_steps, dt, t0)
  # cosine, _dt, _t0, _t1 = get_cosine_beta_schedule(num_steps, dt, t0, offset)
  # linear = linear.flatten()
  # cosine = cosine.flatten()
  # # print(linear)
  # # print(cosine)
  # # print(SSlinear)
  # # print(SScosine)

  # import matplotlib.pyplot as plt

  # plt.plot(ii, SSlinear)
  # plt.savefig("SSlinear.png")
  # plt.close()

  # plt.plot(ii, SScosine)
  # plt.savefig("SScosine.png")
  # plt.close()

  # plt.plot(ii, linear)
  # plt.savefig("linear.png")
  # plt.close()

  # plt.plot(ii, cosine)
  # plt.savefig("cosine.png")
  # plt.close()

  # betas = get_betas(num_steps=1000)
  # print(betas)
