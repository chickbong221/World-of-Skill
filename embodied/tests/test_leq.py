import jax
import jax.numpy as jnp
import numpy as np

from embodied.jax.leq import lambda_return


def test_action_gradients_match_leq_jacobian():
  # leq.train gets d ret[t] / d a[t] (later actions following the policy) from
  # one backward pass of ret[0] divided by the coefficient prefix; LEQ takes
  # the diagonal of the full Jacobian. Compare both on a toy model.
  H, N, S, A, lam = 4, 3, 5, 2, 0.95
  ks = jax.random.split(jax.random.PRNGKey(0), 7)
  W = jax.random.normal(ks[0], (S, S)) * 0.5
  U = jax.random.normal(ks[1], (A, S)) * 0.5
  P = jax.random.normal(ks[2], (S, A)) * 0.5
  vr = jax.random.normal(ks[3], (S,))
  vq = jax.random.normal(ks[4], (S + A,))
  s0 = jax.random.normal(ks[5], (N, S))
  live = 0.9 + 0.1 * jax.random.uniform(ks[6], (H, N))
  mask = jnp.cumprod(jnp.concatenate([jnp.ones((1, N)), live / 0.997]), 0)

  def rets(delta):
    s, qs, rs = s0, [], []
    for t in range(H + 1):
      a = jnp.tanh(s @ P) + delta[t]
      qs.append(jnp.concatenate([s, a], -1) @ vq)
      if t < H:
        s = jnp.tanh(s @ W + a @ U)
        rs.append(s @ vr)
    return lambda_return(jnp.stack(qs), jnp.stack(rs), live, mask, lam)

  zeros = jnp.zeros((H + 1, N, A))
  jac = jax.jacrev(lambda d: rets(d)[0].sum(1))(zeros)
  ref = jnp.stack([jac[t, t] for t in range(H + 1)])
  grad = jax.grad(lambda d: rets(d)[0][0].sum())(zeros)
  prefix = jnp.cumprod(jnp.concatenate([jnp.ones((1, N)), rets(zeros)[1]]), 0)
  np.testing.assert_allclose(
      grad / prefix[..., None], ref, rtol=1e-4, atol=1e-6)
