"""MoSS: Mixture of Specialized Skills for Subset-Shared World Modeling.

Drop-in replacement for `rssm.RSSM` that routes the deterministic recurrent
update through a sparse mixture of specialized experts, estimates per-
environment predictive responsibility, and applies transition-level subset
invariance (MMD) only between environments an expert is jointly responsible
for.

Public API is identical to `rssm.RSSM` (`initial`, `truncate`, `starts`,
`observe`, `imagine`, `loss`, `entry_space`) except that `loss` accepts the
extra keyword arguments `task_id`, `reward` and `cont`.

Design notes (deviations from the paper text, forced by the real RSSM):

* The paper writes an expert-level L2 loss ||h_hat^(m) - sg(h_{t+1})||^2. In
  an RSSM there is no observation-conditioned `h_{t+1}` target: `deter` is
  *produced* by the core itself. Supervising experts to match the mixture's
  own output is degenerate and would suppress specialization. The correct
  analogue is a per-expert *prior KL*: each active expert's `deter_m` must
  independently explain the posterior over `z_t`. We use that both as the
  expert loss (`mexp`) and as the predictive-quality term in the
  responsibility statistic.
* The stochastic prior head is *shared* across experts (experts specialize the
  action-conditioned transition; the categorical prior stays closed-form).
* Transition signatures are randomly projected before the MMD, since `deter`
  is high dimensional and raw kernels would be both meaningless and huge.
"""

import einops
import elements
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


# Single source of truth for the MoSS loss keys and their default scales.
# `Agent.loss` asserts set(losses) == set(scales) exactly, so these MUST stay in
# sync with the keys written into `losses` in MoSSRSSM.loss.
DEFAULT_SCALES = dict(
    mexp=0.5,     # per-expert prior KL (every active expert must explain z_t)
    msub=1.0,     # transition-level subset invariance (MMD)
    mdiv=0.01,    # expert diversity
    msp=0.001,    # router entropy (confident routing)
    mbal=0.01,    # load balancing
    mrew=1.0,     # per-expert reward head   (only if expert_heads)
    mcon=1.0,     # per-expert continuation head (only if expert_heads)
)


def loss_scales(config_scales=None, expert_heads=True):
  """Return the exact set of MoSS loss scales, defaults filled in."""
  scales = dict(DEFAULT_SCALES)
  scales.update(dict(config_scales or {}))
  if not expert_heads:
    scales.pop('mrew', None)
    scales.pop('mcon', None)
  return scales


class MoSSRSSM(nj.Module):

  # --- Backbone (mirrors rssm.RSSM) ---
  deter: int = 1024          # keep modest: experts replicate the recurrent core
  hidden: int = 512
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

  # --- MoSS ---
  experts: int = 8           # M
  topk: int = 2              # k
  zdim: int = 128            # expert internal representation dim
  expert_heads: bool = True  # per-expert reward / continuation predictors
  router_noise: bool = True  # noisy top-k gating (Shazeer et al., 2017)

  # Responsibility
  num_envs: int = 16         # K (number of training environments/tasks)
  ema: float = 0.99          # beta_ema
  tau_resp: float = 1.0      # tau_resp
  nmin: int = 256            # N_min, accumulated activation support gate

  # Subset invariance
  sigdim: int = 32           # random-projection dim for h and delta-h blocks
  maxsig: int = 32           # S: signatures kept per (env, expert) per batch
  nsig: int = 8              # N_sig, min in-batch signatures per env
  warm_steps: int = 5000     # T_warm  (also: dense routing during this window)
  ramp_steps: int = 5000     # T_ramp
  bal_impl: str = 'switch'   # 'switch' (recommended) or 'paper'

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    assert self.topk <= self.experts
    self.act_space = act_space
    self.kw = kw
    # Fixed (non-trained) random projection for MMD signatures. Deterministic
    # across runs; baked into the jit graph as a constant.
    rng = np.random.default_rng(0)
    self._proj = (rng.normal(size=(self.deter, self.sigdim))
                  / np.sqrt(self.deter)).astype(np.float32)
    # Non-gradient responsibility state, shape (K, M).
    K, M = self.num_envs, self.experts
    self.q_ema = nj.Variable(jnp.zeros, (K, M), f32, name='q_ema')
    self.l_ema = nj.Variable(jnp.zeros, (K, M), f32, name='l_ema')
    self.count = nj.Variable(jnp.zeros, (K, M), f32, name='count')
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

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
    carry, (entries, feat) = nj.scan(
        lambda carry, inputs: self._observe(carry, *inputs, training),
        carry, (tokens, action, reset), unroll=unroll, axis=1)
    return carry, entries, feat

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      # Routing noise is disabled outside of observation/training.
      deter, ex = self._core(
          carry['deter'], carry['stoch'], actemb, noise=False)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(
          deter=deter, stoch=stoch, logit=logit,
          pi=ex['pi'], pitilde=ex['pitilde'], delta=ex['delta'],
          zexp=ex['zexp'], sig=ex['sig'],
          klexp=jnp.zeros(ex['pi'].shape, ex['pi'].dtype)))
      if self.expert_heads:
        # Key set MUST match _observe: the agent tree-concats repfeat+imgfeat.
        feat['rexp'] = ex['rexp']
        feat['cexp'] = ex['cexp']
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

  # ------------------------------------------------------------- internals

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)

    deter, ex = self._core(deter, stoch, action, noise=training)

    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    # Per-expert prior KL: each ACTIVE expert must independently explain the
    # posterior over z_t. This replaces the paper's L2-on-h_{t+1}, which has
    # no analogue in an RSSM (see module docstring).
    post = self._dist(sg(logit))
    klexp = []
    for m in range(self.experts):
      prior_m = self._prior(ex['deters'][m])
      klexp.append(post.kl(self._dist(prior_m)))
    klexp = jnp.stack(klexp, -1)                       # (..., M)

    carry = dict(deter=deter, stoch=stoch)
    feat = dict(
        deter=deter, stoch=stoch, logit=logit,
        pi=ex['pi'], pitilde=ex['pitilde'], delta=ex['delta'],
        zexp=ex['zexp'], sig=ex['sig'], klexp=nn.cast(klexp))
    if self.expert_heads:
      feat['rexp'] = ex['rexp']
      feat['cexp'] = ex['cexp']
    entry = dict(deter=deter, stoch=stoch)
    return carry, (entry, feat)

  def _router(self, x, noise):
    """Noisy top-k gating. Returns full pi, sparse-normalized pitilde, delta."""
    logits = self.sub('rlogit', nn.Linear, self.experts, **self.kw)(x)
    if self.router_noise and noise:
      scale = jax.nn.softplus(
          self.sub('rnoise', nn.Linear, self.experts, **self.kw)(x))
      eps = jax.random.normal(nj.seed(), logits.shape, logits.dtype)
      logits = logits + eps * scale
    pi = jax.nn.softmax(f32(logits), -1)

    _, idx = jax.lax.top_k(f32(logits), self.topk)     # (..., k)
    delta = jax.nn.one_hot(idx, self.experts, dtype=f32).sum(-2)

    # Dense routing during warm-up so every expert receives supervision and the
    # responsibility statistics can stabilize before subset alignment starts.
    dense = (self.step.read() < self.warm_steps).astype(f32)
    delta = dense * jnp.ones_like(delta) + (1 - dense) * delta

    masked = pi * delta
    pitilde = masked / jnp.maximum(masked.sum(-1, keepdims=True), 1e-8)
    return nn.cast(pi), nn.cast(pitilde), nn.cast(delta)

  def _core(self, deter, stoch, action, noise):
    """Routed recurrent update. Each expert owns its own block-GRU."""
    stoch = stoch.reshape((stoch.shape[0], -1))
    action = action / sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

    # Shared input projections (as in DreamerV3).
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    shared = jnp.concatenate([x0, x1, x2], -1)

    pi, pitilde, delta = self._router(sg(shared), noise)

    deters, zexps = [], []
    for m in range(self.experts):
      # Expert internal representation z^(m) = E_m(.)
      z = self.sub(f'e{m}in', nn.Linear, self.zdim, **self.kw)(shared)
      z = nn.act(self.act)(self.sub(f'e{m}innorm', nn.Norm, self.norm)(z))
      zexps.append(z)
      # Expert-specific block-GRU over the common deterministic state.
      h = z[..., None, :].repeat(g, -2)
      h = group2flat(jnp.concatenate([flat2group(deter), h], -1))
      for i in range(self.dynlayers):
        h = self.sub(f'e{m}hid{i}', nn.BlockLinear, self.deter, g, **self.kw)(h)
        h = nn.act(self.act)(
            self.sub(f'e{m}hid{i}norm', nn.Norm, self.norm)(h))
      h = self.sub(f'e{m}gru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(h)
      reset, cand, update = [group2flat(y) for y in jnp.split(flat2group(h), 3, -1)]
      reset = jax.nn.sigmoid(reset)
      cand = jnp.tanh(reset * cand)
      update = jax.nn.sigmoid(update - 1)
      deters.append(update * cand + (1 - update) * deter)

    stacked = jnp.stack(deters, -2)                    # (..., M, deter)
    zexp = jnp.stack(zexps, -2)                        # (..., M, zdim)
    w = nn.cast(pitilde)[..., None]
    routed = (w * stacked).sum(-2)                     # (..., deter)

    # Transition signatures: [P h, a, P dh^(m)], blocks normalized separately.
    proj = jnp.asarray(self._proj, stacked.dtype)
    dh = stacked - deter[..., None, :]                 # (..., M, deter)
    hs = (sg(deter) @ proj)[..., None, :].repeat(self.experts, -2)
    ds = dh @ proj
    a_ = action[..., None, :].repeat(self.experts, -2)
    blk = lambda x: x / sg(jnp.maximum(x.std(), 1e-4))
    sig = jnp.concatenate([blk(hs), blk(a_), blk(ds)], -1)

    ex = dict(
        pi=pi, pitilde=pitilde, delta=delta,
        deters=deters, zexp=zexp, sig=nn.cast(sig))
    if self.expert_heads:
      inp = jnp.concatenate(
          [zexp, action[..., None, :].repeat(self.experts, -2)], -1)
      r = self.sub('erew', nn.Linear, 1, **self.kw)(inp)[..., 0]
      c = self.sub('econ', nn.Linear, 1, **self.kw)(inp)[..., 0]
      ex['rexp'], ex['cexp'] = nn.cast(r), nn.cast(c)
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

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    return embodied.jax.outs.Agg(out, 1, jnp.sum)

  # ----------------------------------------------------------------- loss

  def loss(self, carry, tokens, acts, reset, training,
           task_id=None, reward=None, cont=None):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    B, T = reset.shape

    # Standard RSSM KLs on the ROUTED prediction.
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}

    pi = f32(feat['pi'])              # (B,T,M)
    pit = f32(feat['pitilde'])
    delta = f32(feat['delta'])
    klexp = f32(feat['klexp'])        # (B,T,M)
    M = self.experts

    # ---- Expert-level dynamics: every active expert must explain z_t.
    losses['mexp'] = (delta * sg(pit) * klexp).sum(-1)

    # ---- Per-expert reward / continuation heads (routed composition).
    if self.expert_heads and reward is not None:
      rhat = (pit * f32(feat['rexp'])).sum(-1)
      losses['mrew'] = jnp.square(nn.symlog(f32(reward)) - rhat)
      chat = (pit * f32(feat['cexp'])).sum(-1)
      losses['mcon'] = optax_bce(chat, f32(cont))

    # ---- Routing regularizers.
    ent = -(pi * jnp.log(pi + 1e-8)).sum(-1)
    losses['msp'] = ent
    pbar = pi.mean((0, 1))                             # (M,)
    if self.bal_impl == 'switch':
      fbar = delta.mean((0, 1)) * (M / max(self.topk, 1))
      bal = M * (fbar * pbar).sum()
    else:
      bal = jnp.square(pbar - 1.0 / M).sum()
    losses['mbal'] = jnp.broadcast_to(bal, (B, T))

    # ---- Expert diversity on masked internal representations.
    z = f32(feat['zexp']) * delta[..., None]           # (B,T,M,zdim)
    z = z.reshape((B * T, M, self.zdim)).transpose((1, 0, 2))   # (M,N,zdim)
    z = z / (jnp.linalg.norm(z, axis=(1, 2), keepdims=True) + 1e-8)
    gram = jnp.einsum('mnd,pne->mpde', z, z) / (B * T)
    off = 1.0 - jnp.eye(M)
    div = (jnp.square(gram).sum((-1, -2)) * off).sum() / max(M * (M - 1), 1)
    losses['mdiv'] = jnp.broadcast_to(div, (B, T))

    # ---- Responsibility + transition-level subset invariance.
    if task_id is None:
      losses['msub'] = jnp.zeros((B, T), f32)
    else:
      env = task_id.reshape(-1).astype(i32)            # (N,)
      sub, rmets = self._subset_loss(
          env, delta.reshape((-1, M)), pit.reshape((-1, M)),
          klexp.reshape((-1, M)),
          f32(feat['sig']).reshape((-1, M, feat['sig'].shape[-1])),
          training)
      losses['msub'] = jnp.broadcast_to(sub, (B, T))
      metrics.update(rmets)

    if training:
      self.step.write(self.step.read() + 1.0)

    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    metrics['moss_usage'] = delta.mean()
    metrics['moss_pi_max'] = pi.max(-1).mean()
    metrics['moss_ent'] = ent.mean()
    metrics['moss_lambda'] = self._lambda_sub()
    return carry, entries, losses, feat, metrics

  def _lambda_sub(self):
    s = self.step.read()
    return jnp.clip((s - self.warm_steps) / max(self.ramp_steps, 1), 0.0, 1.0)

  def _subset_loss(self, env, delta, pit, klexp, sig, training):
    """Responsibility EMAs + responsibility-gated pairwise MMD."""
    K, M, S = self.num_envs, self.experts, self.maxsig
    N = env.shape[0]
    onehot = jax.nn.one_hot(env, K, dtype=f32)         # (N,K)
    envcnt = onehot.sum(0)                             # (K,)

    # --- Batch statistics, Eq. routing-mass / expert-error.
    aw = delta * pit                                   # (N,M) activation weight
    q_b = jnp.einsum('nk,nm->km', onehot, aw) / jnp.maximum(
        envcnt[:, None], 1.0)
    num = jnp.einsum('nk,nm->km', onehot, aw * klexp)
    den = jnp.einsum('nk,nm->km', onehot, aw)
    l_b = num / (den + 1e-8)
    supp = jnp.einsum('nk,nm->km', onehot, delta)      # (K,M) in-batch support
    seen = (supp > 0).astype(f32)

    # --- EMA updates (detached, only when training).
    if training:
      axes = internal.get_data_axes()
      if axes:
        q_b = jax.lax.pmean(q_b, axes)
        l_b = jax.lax.pmean(l_b, axes)
        supp = jax.lax.psum(supp, axes)
        seen = (supp > 0).astype(f32)
      b = self.ema
      self.q_ema.write(sg(
          b * self.q_ema.read() + (1 - b) * q_b))
      self.l_ema.write(sg(
          seen * (b * self.l_ema.read() + (1 - b) * l_b)
          + (1 - seen) * self.l_ema.read()))
      self.count.write(sg(self.count.read() + supp))

    q = sg(self.q_ema.read())
    l = sg(self.l_ema.read())
    cnt = sg(self.count.read())

    # --- Predictive responsibility (softmax-normalized over experts).
    s_em = q * jnp.exp(-l / self.tau_resp)
    rho = s_em / (s_em.sum(-1, keepdims=True) + 1e-8)  # (K,M)
    chi = (cnt >= self.nmin).astype(f32)               # (K,M) support gate
    # Pairwise weight a_ij^(m) = chi_i chi_j sqrt(rho_i rho_j).
    a = jnp.sqrt(
        jnp.maximum(rho[:, None, :] * rho[None, :, :], 0.0) + 1e-12)
    a = a * chi[:, None, :] * chi[None, :, :]          # (K,K,M)
    a = sg(a)

    # --- Fixed-size signature sampling per (env, expert): static shapes.
    key = jax.random.uniform(nj.seed(), (N,))
    mmd = []
    for m in range(M):
      # score>0 only for transitions of env e where expert m is active.
      score = onehot * (delta[:, m] * key)[:, None]    # (N,K)
      val, idx = jax.lax.top_k(score.T, S)             # (K,S)
      valid = (val > 0).astype(f32)                    # (K,S)
      x = sig[:, m, :][idx]                            # (K,S,D)
      D = x.shape[-1]
      x = x.reshape((K * S, D))
      w = valid.reshape((K * S,))
      # Gate environments with too few in-batch signatures.
      nb = valid.sum(-1)                               # (K,)
      ok = (nb >= self.nsig).astype(f32)               # (K,)
      w = w * jnp.repeat(ok, S)

      d2 = jnp.maximum(
          jnp.sum(x ** 2, -1)[:, None] + jnp.sum(x ** 2, -1)[None, :]
          - 2 * x @ x.T, 0.0)
      med = sg(jnp.maximum(
          jnp.sum(d2 * (w[:, None] * w[None, :]))
          / (jnp.sum(w[:, None] * w[None, :]) + 1e-8), 1e-6))
      kmat = sum(jnp.exp(-d2 / (2 * (b * med) + 1e-8))
                 for b in (0.5, 1.0, 2.0, 4.0))

      W = (jnp.repeat(jnp.eye(K), S, axis=0) * w[:, None])   # (K*S, K)
      Ssum = W.T @ kmat @ W                            # (K,K) kernel sums
      n = W.sum(0)                                     # (K,)
      mean = Ssum / (n[:, None] * n[None, :] + 1e-8)
      dg = jnp.diag(mean)
      mmd.append(jnp.maximum(dg[:, None] + dg[None, :] - 2 * mean, 0.0))
    mmd = jnp.stack(mmd, -1)                           # (K,K,M)

    # Upper-triangular env pairs only.
    tri = jnp.triu(jnp.ones((K, K)), 1)[..., None]
    w_ij = a * tri
    loss = (w_ij * mmd).sum() / (w_ij.sum() + 1e-8)
    loss = self._lambda_sub() * loss

    mets = {
        'moss_rho_max': rho.max(-1).mean(),
        'moss_rho_ent': -(rho * jnp.log(rho + 1e-8)).sum(-1).mean(),
        'moss_pairs': (w_ij > 0).astype(f32).sum(),
        'moss_mmd': (mmd * tri).sum() / (tri.sum() * M + 1e-8),
        'moss_support': (chi.mean()),
    }
    return loss, mets


def optax_bce(logit, target):
  # Numerically stable BCE-with-logits.
  return jnp.maximum(logit, 0) - logit * target + jnp.log1p(
      jnp.exp(-jnp.abs(logit)))