"""Calculate the normalising constant using CMCD."""
import jax
from absl import app, flags
from ml_collections.config_flags import config_flags
import wandb
import pickle
import haiku as hk
import distrax
import jax.numpy as jnp
from typing import Optional
import chex
from cmcd.run_lib import (
  training,
  sample_from_target,
  setup_training,
  )
from diffusionjax.utils import flatten_nested_dict

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", './configs/nice.py', "Training configuration.",
  lock_config=True)
flags.DEFINE_string("workdir", './examples/', "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

Array = jax.Array


class NICE(hk.Module):
  """Implements a NICE flow."""

  def __init__(
      self,
      dim: int,
      n_steps: int = 4,
      h_depth: int = 5,
      h_dim: int = 1000,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    self._dim = dim
    self._half_dim = dim // 2
    self._nets = []
    for _ in range(n_steps):
      layers = []
      for j in range(h_depth):
        if j != h_depth - 1:
          layers.append(hk.Linear(h_dim))
          layers.append(jax.nn.relu)
        else:
          layers.append(hk.Linear(self._half_dim))
      net = hk.Sequential(layers)
      self._nets.append(net)

    self._parts = []
    self._inv_parts = []
    for _ in range(n_steps):
      shuff = list(reversed(range(dim)))
      self._parts.append(shuff)
      self._inv_parts.append(shuff)

    self._logscale = hk.get_parameter("logscale", (dim,), init=jnp.zeros)

  def forward(self, x: Array) -> Array:
    """Runs the model x->y."""
    chex.assert_shape(x, (None, self._dim))

    split = self._half_dim
    if self._dim % 2 == 1:
      split += 1

    for part, net in zip(self._parts, self._nets):
      x_shuff = x[:, part]
      xa, xb = x_shuff[:, :split], x_shuff[:, split:]
      ya = xa
      yb = xb + net(xa)
      x = jnp.concatenate([ya, yb], -1)

    chex.assert_shape(x, (None, self._dim))
    return x

  def reverse(self, y: Array) -> Array:
    """Runs the model y->x."""
    chex.assert_shape(y, (None, self._dim))

    split = self._half_dim
    if self._dim % 2 == 1:
      split += 1

    for inv_part, net in reversed(list(zip(self._inv_parts, self._nets))):
      ya, yb = y[:, :split], y[:, split:]
      xa = ya
      xb = yb - net(xa)
      x_shuff = jnp.concatenate([xa, xb], -1)
      y = x_shuff[:, inv_part]

    chex.assert_shape(y, (None, self._dim))
    return y

  def logpx(self, x: Array) -> Array:
    """Rreturns logp(x)."""
    z = self.forward(x)
    zs = z * jnp.exp(self._logscale)[None, :]

    pz = distrax.MultivariateNormalDiag(jnp.zeros_like(zs), jnp.ones_like(zs))
    logp = pz.log_prob(zs)
    logp = logp + self._logscale.sum()

    chex.assert_shape(logp, (x.shape[0],))
    return logp

  def sample(self, n: int) -> Array:
    """Draws n samples from model."""
    zs = jax.random.normal(hk.next_rng_key(), (n, self._dim))
    z = zs / jnp.exp(self._logscale)[None, :]
    x = self.reverse(z)

    chex.assert_shape(x, (n, self._dim))
    return x

  def reparameterized_sample(self, zs: Array) -> Array:
    """Draws n samples from model."""
    z = zs / jnp.exp(self._logscale)[None, :]
    x = self.reverse(z)

    chex.assert_shape(x, zs.shape)
    return x

  def loss(self, x: Array) -> Array:
    """Loss function for training."""
    return -self.logpx(x)


def load_model(config):
  artifact_name = f"{config.data.alpha}_{config.data.n_bits}_{config.data.im_size}"
  api = wandb.Api()
  artifact = api.artifact(f"shreyaspadhy/cais/{artifact_name}:latest")
  loaded_params = pickle.load(open(artifact.file(), "rb"))

  def forward_fn():
    flow = NICE(config.data.im_size**2, h_dim=config.data.hidden_dim)

    def _logpx(x):
      return flow.logpx(x)

    def _recons(x):
      return flow.reverse(flow.forward(x))

    def _sample(n):
      return flow.sample(n)

    return _logpx, (_logpx, _recons, _sample)

  forward = hk.multi_transform(forward_fn)
  logpx_fn, _, sample_fn = forward.apply
  logpx_fn_without_rng = lambda x: jnp.squeeze(
      logpx_fn(loaded_params, jax.random.PRNGKey(1), x[None, :])
  )
  sample_fn_clean = lambda rng, n: sample_fn(loaded_params, rng, n)
  return logpx_fn_without_rng, config.data.im_size**2, sample_fn_clean


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  # num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

  # jax_config.update("jax_traceback_filtering", "off")
  # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

  log_prob_model, dim, sample_from_target_fn = load_model(config)
  sample_shape = (dim,)

  wandb_kwargs = {
    "project": config.wandb.project,
    "entity": config.wandb.entity,
    "config": flatten_nested_dict(config.to_dict()),
    "name": config.wandb.name if config.wandb.name else None,
    "mode": "online" if config.wandb.log else "disabled",
    "settings": wandb.Settings(code_dir=config.wandb.code_dir),
  }

  with wandb.init(**wandb_kwargs) as run:
    setup_training(run)

    params, samples, target_samples, n_samples = training(
      config, log_prob_model, sample_from_target_fn, sample_shape)
    sample_from_target(config, sample_from_target_fn, samples, target_samples, n_samples)
    make_grid(
      samples, config.im_size, n=64, wandb_prefix="images/final_sample"
    )

    if config.training.use_ema:
      make_grid(
        samples_ema,
        config.im_size,
        n=64,
        wandb_prefix="images/final_sample_ema",
      )

    if config.wandb.log_artifact:
      artifact_name = f"nice_{config.data.alpha}_{config.data.n_bits}_{config.data.im_size}_{config.solver.outer_solver}_{config.solver.num_outer_steps}"
      artifact = wandb.Artifact(
        artifact_name,
        type="final params",
      )
      # Save model
      with artifact.new_file("params.pkl", "wb") as f:
        pickle.dump(params, f)

      wandb.log_artifact(artifact)


if __name__ == "__main__":
    # os.environ["WANDB_API_KEY"] = "9835d6db89010f73306f92bb9a080c9751b25d28"

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    app.run(main)
