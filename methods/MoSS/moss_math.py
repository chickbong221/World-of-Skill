"""Pure functions behind the revised MoSS losses (no ninjax state).

Kept separate from `moss.py` so they can be unit-tested on plain arrays
(embodied/tests/test_moss_math.py). Every function is shape-polymorphic over
leading batch dimensions unless stated otherwise; the expert axis is last.

Paper mapping (revised draft):
  topk_mask, renorm         noise-free / noisy active sets, Eq. (5)-(6)
  competitive_router_loss   L_comp, Eq. (moss_competitive_router)
  hinged_route_mi           L_MI with hinge, Eq. (moss_mi_router)
  route_mi                  plain -I(task; expert) (legacy L_sep)
  router_z_loss             L_z, Eq. (moss_router_z)
  block_mmd                 weighted unbiased MMD, Eq. (expert_pairwise_mmd)
  disagreement              dis_j, Eq. (moss_uncertainty_penalty)
  linear_anneal             sigma(s), Eq. (subset_weight_schedule)
"""

import jax
import jax.numpy as jnp

f32 = jnp.float32
sg = jax.lax.stop_gradient
NEG = -1e30  # finite "minus infinity" for masked logsumexp (NaN-free)


def topk_mask(logits, k):
  """0/1 mask (..., M) of the k largest entries along the last axis."""
  _, idx = jax.lax.top_k(logits, k)
  return jax.nn.one_hot(idx, logits.shape[-1], dtype=f32).sum(-2)


def renorm(probs, mask, eps=1e-8):
  """probs restricted to mask and renormalized to sum to one (Eq. 6)."""
  masked = probs * mask
  return masked / jnp.maximum(masked.sum(-1, keepdims=True), eps)


def _masked_lse(x, mask):
  return jax.nn.logsumexp(jnp.where(mask > 0, x, NEG), axis=-1)


def competitive_router_loss(logits, mask, err, tau):
  """Competitive mixture-of-experts loss on the router (Jacobs et al., 1991).

    -log sum_{m in mask} pi~_m exp(-err_m / tau),
    pi~ = softmax(logits) renormalized over `mask`.

  Computed as LSE_mask(logits - err/tau) - LSE_mask(logits), so it is exact
  and stable. `err` is detached here: the term trains the router only. Its
  gradient w.r.t. logit m (in mask) is pi~_m - post_m with
  post_m ∝ pi~_m exp(-err_m / tau), i.e. it pulls the router toward the
  per-transition posterior over experts. Returns per-row losses (...,).
  """
  logits = f32(logits)
  err = sg(f32(err))
  return -(_masked_lse(logits - err / tau, mask) - _masked_lse(logits, mask))


def router_posterior(logits, mask, err, tau):
  """The posterior pi* that competitive_router_loss pulls the router toward."""
  x = jnp.where(mask > 0, f32(logits) - sg(f32(err)) / tau, NEG)
  return jax.nn.softmax(x, -1)


def router_z_loss(logits):
  """ST-MoE router z-loss: (logsumexp of the clean logits)^2, per row."""
  return jnp.square(jax.nn.logsumexp(f32(logits), axis=-1))


def _group_rows(sums, cnt, eps):
  W = sums / jnp.maximum(cnt[:, None], 1.0)
  W = W / jnp.maximum(W.sum(-1, keepdims=True), eps)      # rows sum to 1
  pg = cnt / jnp.maximum(cnt.sum(), 1.0)
  return W, pg


def route_mi(sums, cnt, eps=1e-8):
  """I(G;E) = H(E) - H(E|G) in nats, from per-group mass sums (G, M) and
  counts (G,). Absent groups (count 0) get P(g) = 0 and drop out.
  Returns (I, H(E), H(E|G))."""
  W, pg = _group_rows(sums, cnt, eps)
  joint = pg[:, None] * W                                  # P(g, e)
  pe = joint.sum(0)                                        # P(e)
  he = -(pe * jnp.log(jnp.maximum(pe, eps))).sum()
  heg = -(joint * jnp.log(jnp.maximum(W, eps))).sum()
  return he - heg, he, heg


def hinged_route_mi(sums, cnt, target, eps=1e-8):
  """Environment-expert objective with a hinge (revised Eq. moss_mi_router).

    L = -H(P(e)) + sum_g P(g) [H(P(e|g)) - log target]_+

  The first term spreads usage over all experts (replaces load balancing);
  the second concentrates each group on about `target` experts and stops
  there, so it never pushes toward one private expert per group. With
  target = 1 this equals -I(G;E). Use target >= k: a group whose
  transitions all use the same k experts equally has H(P(e|g)) = log k.
  Returns (loss, H(E), sum_g P(g) H(E|g)).
  """
  W, pg = _group_rows(sums, cnt, eps)
  pe = (pg[:, None] * W).sum(0)
  he = -(pe * jnp.log(jnp.maximum(pe, eps))).sum()
  hrow = -(W * jnp.log(jnp.maximum(W, eps))).sum(-1)      # (G,)
  hinge = jnp.maximum(hrow - jnp.log(float(target)), 0.0)
  return -he + (pg * hinge).sum(), he, (pg * hrow).sum()


def block_mmd(kmat, w, K, S, eps=1e-8):
  """Weighted unbiased squared MMD between all pairs of K sample blocks.

  kmat: (K*S, K*S) kernel matrix of samples laid out as K blocks of S rows.
  w:    (K*S,) nonnegative sample weights (0 = unused slot).
  Within-block terms exclude the diagonal (U-statistic); with 0/1 weights
  this is exactly the standard unbiased estimator. Returns (K, K).
  """
  Wm = jnp.repeat(jnp.eye(K, dtype=f32), S, axis=0) * w[:, None]  # (K*S, K)
  Ssum = Wm.T @ kmat @ Wm                                    # (K, K)
  sw = Wm.sum(0)
  sq = jnp.square(Wm).sum(0)
  self_terms = (jnp.square(Wm) * jnp.diag(kmat)[:, None]).sum(0)
  within = (jnp.diag(Ssum) - self_terms) / jnp.maximum(sw * sw - sq, eps)
  cross = Ssum / jnp.maximum(sw[:, None] * sw[None, :], eps)
  return within[:, None] + within[None, :] - 2.0 * cross


def disagreement(stacked, mask, weights=None):
  """Spread of the active experts' candidate states (Eq. 28), per row.

  stacked: (..., M, D) candidate next states (or displacements: same value).
  mask:    (..., M) active experts.
  weights: None for equal weights over the active set (revised draft), or
           routing weights pi~ (legacy).
  """
  if weights is None:
    weights = mask / jnp.maximum(mask.sum(-1, keepdims=True), 1.0)
  stacked = f32(stacked)
  mean = (weights[..., None] * stacked).sum(-2)
  spread = jnp.square(stacked - mean[..., None, :]).mean(-1)
  return (weights * spread).sum(-1)


def linear_anneal(step, start, steps):
  """start at step 0, decreasing linearly to 0 at `steps` (0: constant)."""
  if steps <= 0:
    return f32(start) + 0.0 * step
  return f32(start) * jnp.clip(1.0 - step / steps, 0.0, 1.0)


def ramp(step, steps):
  """0 at step 0, increasing linearly to 1 at `steps` (0: always 1)."""
  if steps <= 0:
    return 1.0 + 0.0 * step
  return jnp.clip(step / steps, 0.0, 1.0)
