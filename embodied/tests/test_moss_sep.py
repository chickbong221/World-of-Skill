import jax
import jax.numpy as jnp
import numpy as np

from methods.MoSS.moss import route_mi

M = 8


def sep_loss(mass, groups, G):
  """-I(G;E) / min(log G, log M) from per-token mass (N, M), as in loss()."""
  onehot = jax.nn.one_hot(groups, G, dtype=jnp.float32)
  sums = onehot.T @ mass
  cnt = onehot.sum(0)
  mi, _, _ = route_mi(sums, cnt)
  return -mi / min(np.log(G), np.log(M))


def tokens(rows, n=10):
  """Per-group mass rows (G, M) -> n identical tokens per group."""
  rows = jnp.asarray(rows, jnp.float32)
  mass = jnp.repeat(rows, n, 0)
  groups = jnp.repeat(jnp.arange(len(rows)), n)
  return mass, groups


def test_disjoint_is_minus_one():
  rows = np.zeros((4, M))
  for g in range(4):
    rows[g, 2 * g: 2 * g + 2] = 0.5
  assert np.isclose(sep_loss(*tokens(rows), 4), -1.0, atol=1e-5)


def test_uniform_is_zero():
  rows = np.full((4, M), 1 / M)
  assert np.isclose(sep_loss(*tokens(rows), 4), 0.0, atol=1e-6)


def test_collapsed_is_zero():
  rows = np.zeros((4, M))
  rows[:, 5], rows[:, 7] = 0.33, 0.67
  assert np.isclose(sep_loss(*tokens(rows), 4), 0.0, atol=1e-6)


def test_absent_groups_ignored():
  rows = np.zeros((2, M))
  rows[0, :2] = rows[1, 2:4] = 0.5
  mass, groups = tokens(rows)
  # Same tokens with G = 4 (groups 2, 3 absent): I unchanged = log 2.
  onehot = jax.nn.one_hot(groups, 4, dtype=jnp.float32)
  mi, he, heg = route_mi(onehot.T @ mass, onehot.sum(0))
  assert np.isclose(mi, np.log(2), atol=1e-5)
  assert np.isclose(he - heg, mi)


def test_gradient_reaches_router_logits():
  rng = np.random.default_rng(0)
  logits = jnp.asarray(rng.normal(size=(40, M)), jnp.float32)
  groups = jnp.repeat(jnp.arange(4), 10)
  fn = lambda z: sep_loss(jax.nn.softmax(z, -1), groups, 4)
  grad = jax.grad(fn)(logits)
  assert np.isfinite(grad).all()
  assert np.abs(grad).sum() > 0
