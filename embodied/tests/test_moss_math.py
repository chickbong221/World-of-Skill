import jax
import jax.numpy as jnp
import numpy as np

from methods.MoSS import moss_math as mm

M, K = 8, 2


def softmax(x):
  e = np.exp(x - x.max(-1, keepdims=True))
  return e / e.sum(-1, keepdims=True)


class TestRouting:

  def test_topk_mask_and_renorm(self):
    logits = jnp.asarray([[3.0, 1.0, 2.0, 0.0]])
    mask = mm.topk_mask(logits, 2)
    np.testing.assert_array_equal(mask, [[1, 0, 1, 0]])
    w = mm.renorm(jax.nn.softmax(logits, -1), mask)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(w[0, [0, 2]], softmax(np.array([3.0, 2.0])),
                               atol=1e-6)

  def test_selection_noise_keeps_clean_weights(self):
    # noise_impl 'select': whatever set the noise selects, its weights are the
    # clean softmax renormalized over it, i.e. what imagination would use for
    # that set.
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(64, M)), jnp.float32)
    noisy = logits + 2.0 * rng.normal(size=(64, M))
    mask = mm.topk_mask(noisy, K)
    w = mm.renorm(jax.nn.softmax(logits, -1), mask)
    ref = np.where(mask > 0, np.exp(np.asarray(logits)), 0.0)
    ref = ref / ref.sum(-1, keepdims=True)
    np.testing.assert_allclose(w, ref, atol=1e-5)

  def test_linear_anneal_and_ramp(self):
    vals = [float(mm.linear_anneal(s, 2.0, 100)) for s in (0, 50, 100, 200)]
    np.testing.assert_allclose(vals, [2.0, 1.0, 0.0, 0.0], atol=1e-6)
    assert float(mm.linear_anneal(500, 2.0, 0)) == 2.0
    np.testing.assert_allclose(
        [float(mm.ramp(s, 10)) for s in (0, 5, 20)], [0.0, 0.5, 1.0])


class TestCompetitive:

  def test_value_matches_definition(self):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, M))
    err = rng.random((5, M)) * 3
    mask = np.zeros((5, M))
    mask[:, :3] = 1
    tau = 0.7
    pit = np.where(mask > 0, softmax(logits), 0.0)
    pit /= pit.sum(-1, keepdims=True)
    ref = -np.log((pit * np.exp(-err / tau)).sum(-1))
    out = mm.competitive_router_loss(
        jnp.asarray(logits), jnp.asarray(mask), jnp.asarray(err), tau)
    np.testing.assert_allclose(out, ref, rtol=1e-5)

  def test_gradient_is_prior_minus_posterior(self):
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(size=(4, M)), jnp.float32)
    err = jnp.asarray(rng.random((4, M)) * 2, jnp.float32)
    mask = jnp.asarray(mm.topk_mask(logits, 3))
    tau = 0.5
    grad = jax.grad(lambda z: mm.competitive_router_loss(
        z, mask, err, tau).sum())(logits)
    prior = mm.renorm(jax.nn.softmax(logits, -1), mask)
    post = mm.router_posterior(logits, mask, err, tau)
    np.testing.assert_allclose(grad, (prior - post) * mask, atol=1e-5)

  def test_no_gradient_to_errors_or_when_experts_agree(self):
    logits = jnp.zeros((2, M))
    mask = jnp.ones((2, M))
    err = jnp.ones((2, M))
    g_err = jax.grad(lambda e: mm.competitive_router_loss(
        logits, mask, e, 1.0).sum())(err)
    np.testing.assert_allclose(g_err, 0.0)
    g = jax.grad(lambda z: mm.competitive_router_loss(
        z, mask, err, 1.0).sum())(logits)
    np.testing.assert_allclose(g, 0.0, atol=1e-6)

  def test_router_moves_toward_the_better_expert(self):
    logits = jnp.zeros((1, M))
    mask = np.zeros((1, M), np.float32)
    mask[0, :2] = 1.0
    err = np.zeros((1, M), np.float32)
    err[0, 1] = 5.0                             # expert 0 fits, 1 does not
    mask, err = jnp.asarray(mask), jnp.asarray(err)
    g = jax.grad(lambda z: mm.competitive_router_loss(
        z, mask, err, 1.0).sum())(logits)
    assert g[0, 0] < 0 < g[0, 1]  # descent raises logit 0, lowers logit 1

  def test_z_loss(self):
    z = mm.router_z_loss(jnp.zeros((3, M)))
    np.testing.assert_allclose(z, np.log(M) ** 2, rtol=1e-6)


class TestRouteMI:

  def rows_to_sums(self, rows, n=10):
    rows = np.asarray(rows, np.float32)
    return jnp.asarray(rows * n), jnp.full((len(rows),), float(n))

  def test_hinge_with_target_one_is_minus_mi(self):
    rng = np.random.default_rng(3)
    rows = rng.random((5, M))
    sums, cnt = self.rows_to_sums(rows)
    mi, _, _ = mm.route_mi(sums, cnt)
    val, _, _ = mm.hinged_route_mi(sums, cnt, 1.0)
    np.testing.assert_allclose(val, -mi, atol=1e-5)

  def test_hinge_does_not_prefer_top1_over_k_experts(self):
    G = 13
    one = np.zeros((G, M))
    one[np.arange(G), np.arange(G) % M] = 1.0
    two = np.zeros((G, M))
    for g in range(G):
      two[g, [g % M, (g + 3) % M]] = 0.5
    l_one, _, _ = mm.hinged_route_mi(*self.rows_to_sums(one), 2.0)
    l_two, _, _ = mm.hinged_route_mi(*self.rows_to_sums(two), 2.0)
    # Plain -I prefers one expert per task; the hinge (target k = 2) does not
    # penalize using k experts, and only -H(E) differs.
    mi_one, _, _ = mm.route_mi(*self.rows_to_sums(one))
    mi_two, _, _ = mm.route_mi(*self.rows_to_sums(two))
    assert -mi_one < -mi_two
    assert float(l_two) <= float(l_one) + 1e-6

  def test_hinge_penalizes_spreading_beyond_target(self):
    spread = np.full((4, M), 1.0 / M)
    val, he, heg = mm.hinged_route_mi(*self.rows_to_sums(spread), 2.0)
    np.testing.assert_allclose(he, np.log(M), atol=1e-5)
    np.testing.assert_allclose(val, -np.log(M) + np.log(M) - np.log(2),
                               atol=1e-5)

  def test_gradient_reaches_logits(self):
    rng = np.random.default_rng(4)
    logits = jnp.asarray(rng.normal(size=(40, M)), jnp.float32)
    groups = jnp.repeat(jnp.arange(4), 10)
    onehot = jax.nn.one_hot(groups, 4)

    def fn(z):
      u = jax.nn.softmax(z, -1) * mm.topk_mask(z, K)
      return mm.hinged_route_mi(onehot.T @ u, onehot.sum(0), 2.0)[0]
    g = jax.grad(fn)(logits)
    assert np.isfinite(g).all() and np.abs(g).sum() > 0


class TestBlockMMD:

  def kernel(self, x):
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    return np.exp(-d2)

  def reference(self, xs):
    """Unbiased MMD^2 between the sample sets in xs."""
    K = len(xs)
    out = np.zeros((K, K))
    for i in range(K):
      for j in range(K):
        kii = self.kernel(np.concatenate([xs[i]]))
        kjj = self.kernel(np.concatenate([xs[j]]))
        ni, nj = len(xs[i]), len(xs[j])
        within_i = (kii.sum() - np.trace(kii)) / (ni * (ni - 1))
        within_j = (kjj.sum() - np.trace(kjj)) / (nj * (nj - 1))
        cross = np.exp(-((xs[i][:, None] - xs[j][None]) ** 2).sum(-1)).mean()
        out[i, j] = within_i + within_j - 2 * cross
    return out

  def test_binary_weights_match_unbiased_estimator(self):
    rng = np.random.default_rng(5)
    S, D = 6, 3
    n = [6, 4, 5]
    x = rng.normal(size=(3 * S, D))
    x[S:2 * S] += 1.0
    w = np.zeros(3 * S)
    for k, nk in enumerate(n):
      w[k * S: k * S + nk] = 1.0
    out = mm.block_mmd(jnp.asarray(self.kernel(x)), jnp.asarray(w), 3, S)
    ref = self.reference([x[k * S: k * S + nk] for k, nk in enumerate(n)])
    np.testing.assert_allclose(out, ref, atol=1e-5)

  def test_weights_are_scale_free_and_empty_blocks_are_finite(self):
    rng = np.random.default_rng(6)
    S = 5
    x = rng.normal(size=(2 * S, 2))
    kmat = jnp.asarray(self.kernel(x))
    w = jnp.asarray(rng.random(2 * S))
    a = mm.block_mmd(kmat, w, 2, S)
    b = mm.block_mmd(kmat, 7.0 * w, 2, S)
    np.testing.assert_allclose(a, b, rtol=1e-5)
    w0 = np.zeros(2 * S, np.float32)
    w0[:S] = 1.0                                # second block empty
    w0 = jnp.asarray(w0)
    out = mm.block_mmd(kmat, w0, 2, S)
    assert np.isfinite(out).all()


class TestDisagreement:

  def test_equal_weights_two_experts(self):
    a = np.array([1.0, 0.0, 2.0])
    b = np.array([0.0, 0.0, 0.0])
    stacked = jnp.asarray(np.stack([a, b, a + 9], 0)[None])
    mask = jnp.asarray([[1.0, 1.0, 0.0]])
    out = mm.disagreement(stacked, mask)
    np.testing.assert_allclose(out, [((a - b) ** 2).sum() / (4 * 3)],
                               rtol=1e-6)

  def test_routed_weights_shrink_with_router_confidence(self):
    stacked = jnp.asarray([[[1.0], [-1.0]]])
    mask = jnp.ones((1, 2))
    even = mm.disagreement(stacked, mask, jnp.asarray([[0.5, 0.5]]))
    skew = mm.disagreement(stacked, mask, jnp.asarray([[0.99, 0.01]]))
    equal = mm.disagreement(stacked, mask)
    assert float(skew[0]) < float(even[0]) == float(equal[0])
