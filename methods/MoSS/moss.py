"""MoSS in DreamerV3 style: a routed mixture-of-experts RSSM.

Drop-in replacement for `rssm.RSSM` (same public API: `initial`, `truncate`,
`starts`, `observe`, `imagine`, `loss`, `entry_space`), plus `unc_penalty`
used by the agent during imagination.

How the paper's components map onto the DreamerV3 world model
--------------------------------------------------------------
Backbone        The DreamerV3 RSSM. Model state s_t = [h_t, z_t]; posterior
                q(z_t | h_t, x_t); ONE shared prior p(z_t | h_t); shared reward,
                continuation and decoder heads on s_t. Trained with the usual
                reconstruction + two-sided KL (free nats).
Experts         The sequence model is routed: each expert owns a block-GRU and
                proposes h_t^(m); the mixture h_t = sum_m pi~_m h_t^(m)
                = h_{t-1} + sum_m pi~_m (h_t^(m) - h_{t-1}).
L_exp           Every active expert must explain the posterior on its own:
                KL[sg q || p(. | h_t^(m))]  (free nats as in L_dyn).
Responsibility  Skill-score error (paper Eq. 8 analogue): the expert's KL
                divided by the KL of a persistence forecast p(. | h_{t-1}),
                i.e. the fraction of the latent surprise it fails to explain.
L_sub           Residual invariance. An RSSM has no observation-conditioned
                deterministic target, so the residual lives in z-space: the
                randomized probability integral transform (PIT) of the
                posterior sample under the routed prior. If the routed prior is
                the true conditional distribution, the PIT is exactly
                Uniform(0,1)^S whatever states/actions an environment visits,
                so responsible environments must have equal residual laws.
                The residual of expert m uses the routed (mixture) prior with
                the other experts detached, so it is correct for k >= 2, and a
                param-cancelling trick keeps L_sub gradients out of the shared
                prior head: only expert m is trained by its pair terms.
Uncertainty     Disagreement of co-active experts (paper Eq. 28), normalised by
                a running per-environment percentile; the agent subtracts the
                hinge penalty from imagined rewards.
"""

import elements
import einops
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from embodied.jax import internal

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


# Single source of truth for the MoSS loss keys. `Agent.loss` asserts
# set(losses) == set(scales), so these MUST match the keys written in `loss`.
DEFAULT_SCALES = dict(
    mexp=0.5,     # per-expert prior KL
    msub=1.0,     # residual (PIT) invariance between responsible env pairs
    mdiv=1.0,     # expert feature decorrelation (batch-invariant)
    msp=0.001,    # routing mass on the active set
    mbal=0.01,    # balanced soft routing mass
    mrew=1.0,     # per-expert reward head        (only if expert_heads)
    mcon=1.0,     # per-expert continuation head  (only if expert_heads)
)


def loss_scales(config_scales=None, expert_heads=False):
  scales = dict(DEFAULT_SCALES)
  scales.update(dict(config_scales or {}))
  if not expert_heads:
    scales.pop('mrew', None)
    scales.pop('mcon', None)
  return scales


class MoSSRSSM(nj.Module):

  # --- Backbone (mirrors rssm.RSSM) ---
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  # --- Experts and routing ---
  experts: int = 8              # M
  topk: int = 2                 # k (>= 2 so normalized weights get gradient)
  zdim: int = 128               # expert feature dim (used by L_div)
  router_noise: bool = True     # noisy top-k gating, off in imagination
  dense_warmup: bool = False    # legacy: all experts active during warm-up
  expert_heads: bool = False    # legacy: per-expert reward/cont heads

  # --- Predictive responsibility ---
  num_envs: int = 16            # K; must be >= number of training tasks
  ema: float = 0.99             # beta_ema
  tau_resp: float = 0.25        # temperature on the skill-score error
  nmin: int = 256               # N_min, cumulative activation burn-in

  # --- Residual invariance ---
  maxsig: int = 32              # residuals sampled per (env, expert)
  nsig: int = 8                 # N_sig, min in-batch residuals per env
  bandwidths: tuple = (0.5, 1.0, 2.0, 4.0)
  warm_steps: int = 5000        # T_warm
  ramp_steps: int = 5000        # T_ramp
  bal_impl: str = 'paper'       # 'paper' (soft mass) or 'switch'

  # --- Offline uncertainty penalty (read by the agent) ---
  beta_unc: float = 1.0
  unc_quantile: float = 95.0

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    assert 1 <= self.topk <= self.experts
    self.act_space = act_space
    self.kw = kw
    K, M = self.num_envs, self.experts
    # Detached statistics (zero gradient; never touched by the optimizer).
    self.q_ema = nj.Variable(jnp.zeros, (K, M), f32, name='q_ema')
    self.l_ema = nj.Variable(jnp.zeros, (K, M), f32, name='l_ema')
    self.count = nj.Variable(jnp.zeros, (K, M), f32, name='count')
    self.eseen = nj.Variable(jnp.zeros, (K,), f32, name='eseen')
    self.dbar = nj.Variable(jnp.zeros, (K,), f32, name='dbar')
    self.step = nj.Variable(jnp.zeros, (), f32, name='step')

  # ------------------------------------------------------------------ API

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    return nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    return jax.tree.map(lambda x: x[:, -1], entries)

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False,
              extra=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat, ext) = self._observe(
          carry, tokens, action, reset, training)
      return (carry, entry, feat, ext) if extra else (carry, entry, feat)
    unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
    carry, (entries, feat, ext) = nj.scan(
        lambda carry, inputs: self._observe(carry, *inputs, training),
        carry, (tokens, action, reset), unroll=unroll, axis=1)
    return (carry, entries, feat, ext) if extra else (carry, entries, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter, ex = self._core(
          carry['deter'], carry['stoch'], actemb, noise=False)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      B, M = deter.shape[0], self.experts
      feat = nn.cast(self._feat(
          deter, stoch, logit, ex,
          klexp=jnp.zeros((B, M), f32), klpers=jnp.zeros((B,), f32)))
      return carry, (feat, action)
    unroll = length if self.unroll else 1
    if callable(policy):
      carry, (feat, action) = nj.scan(
          lambda c, _: self.imagine(c, policy, 1, training, single=True),
          nn.cast(carry), (), length, unroll=unroll, axis=1)
    else:
      carry, (feat, action) = nj.scan(
          lambda c, a: self.imagine(c, a, 1, training, single=True),
          nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
    return carry, feat, action

  def unc_penalty(self, dis, env):
    """Hinge penalty on expert disagreement (paper Eq. 28).

    dis: (N, H) disagreement along imagined rollouts; env: (N,) start env.
    Zero for disagreement typical of the recorded data of that environment.
    """
    dbar = sg(self.dbar.read())[env.astype(i32)]          # (N,)
    valid = (dbar > 0)[:, None]
    ratio = f32(dis) / jnp.where(valid, dbar[:, None], 1.0)
    return self.beta_unc * jnp.where(valid, jnp.maximum(0.0, ratio - 1.0), 0.0)

  # ------------------------------------------------------------ internals

  def _feat(self, deter, stoch, logit, ex, klexp, klpers):
    # Identical key set in observe and imagine (agent tree-concats both).
    feat = dict(
        deter=deter, stoch=stoch, logit=logit,
        pi=ex['pi'], pitilde=ex['pitilde'], delta=ex['delta'],
        zexp=ex['zexp'], dis=ex['dis'],
        klexp=klexp, klpers=klpers)
    if self.expert_heads:
      feat['rexp'] = ex['rexp']
      feat['cexp'] = ex['cexp']
    return feat

  def _observe(self, carry, tokens, action, reset, training):
    prev, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter, ex = self._core(prev, stoch, action, noise=training)

    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    klexp, klpers = self._expert_stats(prev, ex, logit)
    carry = dict(deter=deter, stoch=stoch)
    feat = self._feat(deter, stoch, logit, ex, klexp, klpers)
    entry = dict(deter=deter, stoch=stoch)
    ext = dict(shared=ex['shared'], prev=prev)   # inputs for L_sub only
    return carry, (entry, feat, ext)

  def _expert_stats(self, prev, ex, logit):
    post = self._dist(sg(logit))
    # Each expert as a complete predictor of the posterior (L_exp, Eq. 8 num).
    klexp = jnp.stack([
        post.kl(self._dist(self._prior(d))) for d in ex['deters']], -1)
    # Persistence forecast: the prior of the un-updated state (Eq. 8 denom).
    klpers = post.kl(self._dist(self._prior(sg(prev))))
    return f32(klexp), f32(klpers)

  def _router(self, x, noise):
    logits = self.sub('rlogit', nn.Linear, self.experts, **self.kw)(x)
    if self.router_noise and noise:
      scale = jax.nn.softplus(
          self.sub('rnoise', nn.Linear, self.experts, **self.kw)(x))
      eps = jax.random.normal(nj.seed(), logits.shape, logits.dtype)
      logits = logits + eps * scale
    pi = jax.nn.softmax(f32(logits), -1)
    _, idx = jax.lax.top_k(f32(logits), self.topk)
    delta = jax.nn.one_hot(idx, self.experts, dtype=f32).sum(-2)
    if self.dense_warmup:
      dense = (self.step.read() < self.warm_steps).astype(f32)
      delta = dense + (1 - dense) * delta
    masked = pi * delta
    pitilde = masked / jnp.maximum(masked.sum(-1, keepdims=True), 1e-8)
    return pi, pitilde, delta

  def _stem(self, deter, stoch, action):
    """Shared input projections of (h_{t-1}, z_{t-1}, a_{t-1})."""
    stoch = stoch.reshape((stoch.shape[0], -1))
    action = action / sg(jnp.maximum(1, jnp.abs(action)))
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    return jnp.concatenate([x0, x1, x2], -1), action

  def _expert(self, m, shared, deter):
    """Expert m: read-in E_m and block-GRU proposing h_t^(m)."""
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    z = self.sub(f'e{m}in', nn.Linear, self.zdim, **self.kw)(shared)
    z = nn.act(self.act)(self.sub(f'e{m}innorm', nn.Norm, self.norm)(z))
    h = z[..., None, :].repeat(g, -2)
    h = group2flat(jnp.concatenate([flat2group(deter), h], -1))
    for i in range(self.dynlayers):
      h = self.sub(f'e{m}hid{i}', nn.BlockLinear, self.deter, g, **self.kw)(h)
      h = nn.act(self.act)(self.sub(f'e{m}hid{i}norm', nn.Norm, self.norm)(h))
    h = self.sub(f'e{m}gru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(h)
    reset, cand, update = [
        group2flat(y) for y in jnp.split(flat2group(h), 3, -1)]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    return z, update * cand + (1 - update) * deter

  def _core(self, deter, stoch, action, noise):
    """Routed recurrent update. Each expert owns its own block-GRU."""
    shared, action = self._stem(deter, stoch, action)
    pi, pitilde, delta = self._router(sg(shared), noise)
    zexps, deters = zip(*[
        self._expert(m, shared, deter) for m in range(self.experts)])
    deters = list(deters)
    stacked = jnp.stack(deters, -2)                          # (B, M, D)
    routed = (nn.cast(pitilde)[..., None] * stacked).sum(-2)
    # Co-active disagreement (Eq. 28): weighted spread of candidates.
    spread = jnp.square(f32(stacked) - f32(routed)[..., None, :]).mean(-1)
    dis = (pitilde * spread).sum(-1)                         # (B,)
    ex = dict(pi=pi, pitilde=pitilde, delta=delta, deters=deters,
              zexp=jnp.stack(zexps, -2), dis=dis, shared=shared)
    if self.expert_heads:
      inp = jnp.concatenate(
          [ex['zexp'], action[..., None, :].repeat(self.experts, -2)], -1)
      ex['rexp'] = self.sub('erew', nn.Linear, 1, **self.kw)(inp)[..., 0]
      ex['cexp'] = self.sub('econ', nn.Linear, 1, **self.kw)(inp)[..., 0]
    return routed, ex

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _probs(self, logits):
    p = jax.nn.softmax(f32(logits), -1)
    return (1 - self.unimix) * p + self.unimix / self.classes

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    return embodied.jax.outs.Agg(out, 1, jnp.sum)

  def _pit_residual(self, m, idx, deter, stoch, shared, prev, pit):
    """Randomized-PIT residual of expert m on the sampled transitions idx.

    The expert is re-run on DETACHED inputs (paper: F_m(E_m(sg h), a)), so
    L_sub trains expert m's parameters only: no gradient to the shared stem,
    the router, earlier time steps, or other experts. The routed prior is
    used with the other experts detached, which keeps the residual exactly
    Uniform under a correct mixture for any k.
    """
    _, d = self._expert(m, sg(shared[idx]), sg(prev[idx]))
    h = sg(deter[idx])                                     # routed state
    w = sg(pit[idx, m:m + 1]).astype(d.dtype)
    hbar = h + w * (d - sg(d))                             # == h numerically
    base = self._prior(h)
    lg = self._prior(hbar) - base + sg(base)               # param-cancelling
    p = self._probs(lg)                                    # (n, S, C)
    onehot = sg(f32(stoch[idx]))
    V = jax.random.uniform(nj.seed(), onehot.shape[:-1], f32)
    below = ((jnp.cumsum(p, -1) - p) * onehot).sum(-1)
    pz = (p * onehot).sum(-1)
    return below + V * pz                                  # (n, S) in [0,1]

  def _lambda_sub(self):
    s = self.step.read()
    return jnp.clip((s - self.warm_steps) / max(self.ramp_steps, 1), 0.0, 1.0)

  # ----------------------------------------------------------------- loss

  def loss(self, carry, tokens, acts, reset, training,
           task_id=None, reward=None, cont=None):
    metrics = {}
    carry, entries, feat, ext = self.observe(
        carry, tokens, acts, reset, training, extra=True)
    B, T = reset.shape
    M = self.experts

    # DreamerV3 KLs on the routed prior.
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}

    pi = f32(feat['pi'])
    pit = f32(feat['pitilde'])
    delta = f32(feat['delta'])
    klexp = f32(feat['klexp'])

    # L_exp: every active expert is a complete predictor (free nats as L_dyn).
    kle = jnp.maximum(klexp, self.free_nats) if self.free_nats else klexp
    losses['mexp'] = (delta * sg(pit) * kle).sum(-1)

    if self.expert_heads:
      rhat = (pit * f32(feat['rexp'])).sum(-1)
      losses['mrew'] = jnp.square(nn.symlog(f32(reward)) - rhat)
      chat = (pit * f32(feat['cexp'])).sum(-1)
      losses['mcon'] = _bce_logits(chat, f32(cont))

    # L_sp: routing mass on the active set (Eq. 24), compatible with k >= 2.
    losses['msp'] = 1.0 - (pi * delta).sum(-1)
    pbar = pi.mean((0, 1))
    if self.bal_impl == 'switch':
      # Switch Transformer: f_m = fraction of active slots on expert m (sums
      # to 1), so the loss is 1.0 when balanced (previously M = 8x larger).
      fbar = delta.mean((0, 1)) / max(self.topk, 1)
      bal = M * (fbar * pbar).sum()
    else:
      bal = jnp.square(pbar - 1.0 / M).sum()
    losses['mbal'] = jnp.broadcast_to(bal, (B, T))

    # L_div on masked expert features.
    z = f32(feat['zexp']) * delta[..., None]
    z = z.reshape((B * T, M, self.zdim)).transpose((1, 0, 2))
    # Safe norm: jnp.linalg.norm has a NaN gradient at exactly 0, which
    # happens whenever an expert is unused for a whole batch.
    z = z / jnp.sqrt(jnp.sum(z * z, axis=(1, 2), keepdims=True) + 1e-12)
    # No 1/|B|: Z-bar is already Frobenius-normalized, so the extra factor
    # made L_div ~1e-7 and scale as 1/|B|^2. Now batch-invariant in [0, 1].
    gram = jnp.einsum('mnd,pne->mpde', z, z)
    off = 1.0 - jnp.eye(M)
    div = (jnp.square(gram).sum((-1, -2)) * off).sum() / max(M * (M - 1), 1)
    losses['mdiv'] = jnp.broadcast_to(div, (B, T))

    lam = self._lambda_sub()                  # lambda used at THIS step
    if task_id is None:
      losses['msub'] = jnp.zeros((B, T), f32)
    else:
      flat = lambda x: x.reshape((B * T, *x.shape[2:]))
      sub, rmets = self._subset_loss(
          task_id.reshape(-1).astype(i32),
          delta.reshape((-1, M)), pit.reshape((-1, M)),
          klexp.reshape((-1, M)), f32(feat['klpers']).reshape(-1),
          f32(feat['dis']).reshape(-1),
          flat(feat['deter']), flat(feat['stoch']),
          flat(ext['shared']), flat(ext['prev']), training)
      losses['msub'] = jnp.broadcast_to(lam * sub, (B, T))
      metrics.update(rmets)

    if training:
      self.step.write(self.step.read() + 1.0)

    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    metrics['moss_usage'] = delta.mean()
    # Collapse monitors: dead experts get zero gradient from predictions and,
    # before the safe-norm fix, a NaN gradient from L_div.
    load = delta.mean((0, 1)) / max(self.topk, 1)            # sums to 1
    metrics['moss_load_max'] = load.max()
    metrics['moss_dead_experts'] = (load == 0).astype(f32).sum()
    metrics['moss_eff_experts'] = jnp.exp(
        -(load * jnp.log(load + 1e-12)).sum())               # 1..M
    metrics['moss_active_mass'] = (pi * delta).sum(-1).mean()
    metrics['moss_lambda'] = lam
    metrics['moss_dis'] = f32(feat['dis']).mean()
    return carry, entries, losses, feat, metrics

  def _subset_loss(self, env, delta, pit, klexp, klpers, dis,
                   deter, stoch, shared, prev, training):
    K, M, S = self.num_envs, self.experts, self.maxsig
    N, D = env.shape[0], self.stoch
    onehot = jax.nn.one_hot(env, K, dtype=f32)              # (N, K)
    envcnt = onehot.sum(0)

    # ---- Minibatch statistics (Eqs. 7-8).
    aw = delta * pit
    q_b = jnp.einsum('nk,nm->km', onehot, aw) / jnp.maximum(envcnt[:, None], 1)
    num = jnp.einsum('nk,nm->km', onehot, aw * klexp)
    den = jnp.einsum('nk,nm->km', onehot, aw * klpers[:, None])
    l_b = num / (den + 1e-6)                                # skill score
    supp = jnp.einsum('nk,nm->km', onehot, delta)
    dis_e = _masked_percentile(dis, onehot, self.unc_quantile)  # (K,)
    present = envcnt > 0

    if training:
      axes = internal.get_data_axes()
      if axes:
        npres = jax.lax.psum(f32(present), axes)
        q_b = jax.lax.psum(q_b * present[:, None], axes) / jnp.maximum(
            npres[:, None], 1)
        dis_e = jax.lax.psum(dis_e * present, axes) / jnp.maximum(npres, 1)
        seenm = jax.lax.psum(f32(supp > 0), axes)
        l_b = jax.lax.psum(l_b * (supp > 0), axes) / jnp.maximum(seenm, 1)
        supp = jax.lax.psum(supp, axes)
        present = npres > 0
      b = self.ema
      first_env = self.eseen.read() == 0
      first_em = self.count.read() == 0
      seen = supp > 0
      ema = lambda old, new, first: jnp.where(first, new, b * old + (1 - b) * new)
      # Usage: updated whenever the env is present; first update sets it.
      self.q_ema.write(sg(jnp.where(
          present[:, None],
          ema(self.q_ema.read(), q_b, first_env[:, None]),
          self.q_ema.read())))
      # Error: updated only where the expert was active.
      self.l_ema.write(sg(jnp.where(
          seen, ema(self.l_ema.read(), l_b, first_em), self.l_ema.read())))
      self.dbar.write(sg(jnp.where(
          present, ema(self.dbar.read(), dis_e, first_env), self.dbar.read())))
      self.count.write(sg(self.count.read() + supp))
      self.eseen.write(sg(self.eseen.read() + f32(present)))

    q, l, cnt = [sg(v.read()) for v in (self.q_ema, self.l_ema, self.count)]

    # ---- Responsibility (Eq. 9), computed as a stable softmax over experts.
    logits = jnp.where(
        q > 0, jnp.log(jnp.maximum(q, 1e-30)) - l / self.tau_resp, -1e30)
    rho = jax.nn.softmax(logits, -1) * (q > 0).any(-1, keepdims=True)
    chi = f32(cnt >= self.nmin)
    a = jnp.sqrt(rho[:, None, :] * rho[None, :, :] + 1e-12)
    a = sg(a * chi[:, None, :] * chi[None, :, :])           # (K, K, M)

    # ---- Residual-invariance MMD per expert (U-statistic, Eqs. 13-14).
    key = jax.random.uniform(nj.seed(), (N,))
    scale = [2.0 * bw * bw * D / 12.0 for bw in self.bandwidths]
    mmds, oks = [], []
    for m in range(M):
      score = onehot * (delta[:, m] * key)[:, None]
      val, idx = jax.lax.top_k(score.T, S)                   # (K, S)
      valid = f32(val > 0)
      ok = f32(valid.sum(-1) >= self.nsig)                    # (K,)
      w = valid.reshape(-1) * jnp.repeat(ok, S)
      x = self._pit_residual(
          m, idx.reshape(-1), deter, stoch, shared, prev, pit)
      sq = jnp.sum(x * x, -1)
      d2 = jnp.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0)
      kmat = sum(jnp.exp(-d2 / s) for s in scale) / len(scale)
      Wm = jnp.repeat(jnp.eye(K), S, axis=0) * w[:, None]    # (K*S, K)
      Ssum = Wm.T @ kmat @ Wm
      n = Wm.sum(0)
      within = (jnp.diag(Ssum) - n) / jnp.maximum(n * (n - 1), 1.0)
      cross = Ssum / jnp.maximum(n[:, None] * n[None, :], 1.0)
      mmds.append(within[:, None] + within[None, :] - 2 * cross)
      oks.append(ok)
    mmd = jnp.stack(mmds, -1)                                 # (K, K, M)
    okm = jnp.stack(oks, -1)
    tri = jnp.triu(jnp.ones((K, K)), 1)[..., None]
    w_ij = a * okm[:, None, :] * okm[None, :, :] * tri        # Eq. 11
    loss = (w_ij * mmd).sum() / (w_ij.sum() + 1e-8)           # Eq. 15

    seen_l = cnt > 0
    mets = {
        'moss_rho_max': rho.max(-1).mean(),
        'moss_rho_ent': -(rho * jnp.log(rho + 1e-8)).sum(-1).mean(),
        'moss_pairs': (w_ij > 0).astype(f32).sum(),
        'moss_mmd_weighted': loss,
        'moss_support': chi.mean(),
        'moss_skill': (l * seen_l).sum() / jnp.maximum(seen_l.sum(), 1),
        'moss_dbar': (self.dbar.read() * (self.eseen.read() > 0)).sum()
                     / jnp.maximum((self.eseen.read() > 0).sum(), 1),
    }
    return loss, mets


def _bce_logits(logit, target):
  return jnp.maximum(logit, 0) - logit * target + jnp.log1p(
      jnp.exp(-jnp.abs(logit)))


def _masked_percentile(x, mask, q):
  """Per-column linear-interpolated percentile of x over rows where mask > 0.

  NaN-free on purpose (masked rows get a large finite sentinel, not NaN), so
  jax.debug_nans does not fire on it. Columns with no rows return 0.
  x: (N,), mask: (N, K) -> (K,)
  """
  n = mask.sum(0)                                             # (K,)
  vals = jnp.where(mask > 0, x[:, None], 1e30)
  srt = jnp.sort(vals, axis=0)                                # valid rows first
  pos = (q / 100.0) * jnp.maximum(n - 1, 0)
  lo = jnp.floor(pos).astype(i32)
  hi = jnp.minimum(lo + 1, jnp.maximum(n - 1, 0).astype(i32))
  frac = pos - lo
  vlo = jnp.take_along_axis(srt, lo[None, :], 0)[0]
  vhi = jnp.take_along_axis(srt, hi[None, :], 0)[0]
  return jnp.where(n > 0, vlo + frac * (vhi - vlo), 0.0)
