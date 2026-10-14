import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

f32 = jnp.float32
sg = jax.lax.stop_gradient


class Net(nj.Module):

  def __init__(self, outs, layers, units, act, norm, outscale=1.0):
    self.outs = outs
    self.kw = dict(layers=layers, units=units, act=act, norm=norm)
    self.outscale = outscale

  def __call__(self, x):
    x = self.sub('mlp', nn.MLP, **self.kw)(x)
    return {k: f32(self.sub(k, nn.Linear, n, outscale=self.outscale)(x))
            for k, n in self.outs.items()}


class RunNorm(nj.Module):
  """Observation mean/std over all training batches seen so far."""

  def __init__(self, dim, limit=10.0, floor=1e-2):
    self.limit, self.floor = limit, floor
    self.mean = nj.Variable(jnp.zeros, (dim,), f32, name='mean')
    self.m2 = nj.Variable(jnp.zeros, (dim,), f32, name='m2')
    self.count = nj.Variable(jnp.zeros, (), f32, name='count')

  def update(self, x):
    n = f32(x.shape[0])
    mean, m2, count = self.mean.read(), self.m2.read(), self.count.read()
    delta, total = x.mean(0) - mean, count + n
    self.mean.write(mean + delta * n / total)
    self.m2.write(
        m2 + ((x - x.mean(0)) ** 2).sum(0) + delta ** 2 * count * n / total)
    self.count.write(total)

  def __call__(self, x):
    std = jnp.sqrt(self.m2.read() / jnp.maximum(self.count.read(), 1.0))
    x = (x - self.mean.read()) / jnp.maximum(std, self.floor)
    return jnp.clip(x, -self.limit, self.limit)


class Agent(embodied.jax.Agent):
  """BC, TD3+BC and IQL on single transitions (batch_length 1 + 1 context)."""

  def __init__(self, obs_space, act_space, config):
    self.obs_space, self.act_space, self.config = obs_space, act_space, config
    c = config
    assert c.algo in ('bc', 'td3bc', 'iql'), c.algo
    adim = act_space['action'].shape[0]
    odim = obs_space['vector'].shape[0]
    net = dict(layers=c.layers, units=c.units, act=c.act, norm=c.norm)
    self.on = RunNorm(odim, name='on') if c.norm_obs else None
    if c.algo == 'td3bc':
      self.pol = Net({'a': adim}, **net, outscale=0.01, name='pol')
      self.slowpol = embodied.jax.SlowModel(
          Net({'a': adim}, **net, outscale=0.01, name='slowpol'),
          source=self.pol, rate=c.tau)
    else:
      self.pol = Net(
          {'mean': adim, 'logstd': adim}, **net, outscale=0.01, name='pol')
    mods = {'popt': [self.pol]}
    if c.algo != 'bc':
      self.q = [Net({'q': 1}, **net, name=f'q{i}') for i in range(2)]
      self.slowq = [embodied.jax.SlowModel(
          Net({'q': 1}, **net, name=f'slowq{i}'), source=q, rate=c.tau)
          for i, q in enumerate(self.q)]
      mods['copt'] = self.q
    if c.algo == 'iql':
      self.v = Net({'v': 1}, **net, name='v')
      mods['vopt'] = [self.v]
    self.count = nj.Variable(jnp.zeros, (), jnp.int32, name='count')
    anneal = lambda lr, n: optax.cosine_decay_schedule(lr, n) if n > 0 else lr
    lrs = dict(popt=(c.actor_lr, c.anneal), copt=(c.critic_lr, 0),
               vopt=(c.value_lr, 0))
    self.opts = {k: embodied.jax.Optimizer(m, optax.chain(
        optax.clip_by_global_norm(c.clip),
        optax.adam(anneal(*lrs[k]))), summary_depth=1, name=k)
        for k, m in mods.items()}

  @property
  def policy_keys(self):
    return '^(on|pol)/'

  @property
  def ext_space(self):
    return {
        'consec': elements.Space(np.int32),
        'stepid': elements.Space(np.uint8, 20),
    }

  def init_policy(self, batch_size):
    return ()

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()

  def policy(self, carry, obs, mode='train'):
    c = self.config
    s = self._norm(obs['vector'])
    if c.algo == 'td3bc':
      act = self._act(self.pol, s)
    else:
      act, logstd = self._gauss(s)
      if mode == 'train' or c.eval_stochastic:
        act += jnp.exp(logstd) * jax.random.normal(nj.seed(), act.shape)
    return carry, {'action': f32(jnp.clip(act, -1.0, 1.0))}, {}

  def train(self, carry, data):
    b = self._batch(data, update=True)
    if nj.creating():
      self._touch(b)
    mets = getattr(self, '_' + self.config.algo)(b)
    self.count.write(self.count.read() + 1)
    return carry, {}, mets

  def report(self, carry, data):
    b = self._batch(data, update=False)
    s, a, w = b['s'], b['a'], b['w']
    wm = lambda x: (w * x).sum() / jnp.maximum(w.sum(), 1.0)
    mean = self._act(self.pol, s) if self.config.algo == 'td3bc' else (
        self._gauss(s)[0])
    mets = {'mse': wm(((mean - b['raw']) ** 2).mean(-1))}
    if self.config.algo != 'td3bc':
      mets['logp'] = wm(self._logp(s, b['raw']))
    if self.config.algo != 'bc':
      mets['q'] = wm(self._q(self.q, s, a).mean(0))
    return carry, mets

  def _batch(self, data, update):
    vec = data['vector']
    assert vec.shape[1] == 2, 'needs batch_length 1 and replay_context 1'
    s, s2 = vec[:, 0], vec[:, 1]
    if self.on:
      update and self.on.update(s)
      s, s2 = self.on(s), self.on(s2)
    # The recorded actions are a bounded expert plus N(0, 1) noise and the env
    # clips them to [-1, 1]: critics see the executed action `a`, while the
    # policies regress the recorded `raw` action (unbiased for the expert).
    # An episode's last row has no stored next obs (record 1 is then a first
    # record of the next episode), so that transition is masked by `w`.
    raw = data['action'][:, 0]
    return dict(
        s=s, s2=s2, a=jnp.clip(raw, -1.0, 1.0), raw=raw,
        r=data['reward'][:, 1], d=f32(data['is_terminal'][:, 0]),
        w=f32(~data['is_first'][:, 1]))

  def _touch(self, b):
    # Create every net before SlowModel copies it.
    s, a = b['s'], b['a']
    self.pol(s)
    if self.config.algo != 'bc':
      self._q(self.q, s, a)
      self._q(self.slowq, s, a)
    if self.config.algo == 'td3bc':
      self.slowpol(s)
    if self.config.algo == 'iql':
      self.v(s)

  def _bc(self, b):
    s, raw, w = b['s'], b['raw'], b['w']
    wm = lambda x: (w * x).sum() / jnp.maximum(w.sum(), 1.0)

    def loss():
      logp = self._logp(s, raw)
      return wm(-logp), {'logp': wm(logp)}

    mets, aux = self.opts['popt'](loss, has_aux=True)
    return {**mets, **aux}

  def _td3bc(self, b):
    c = self.config
    s, a, r, s2, d, w = [b[k] for k in ('s', 'a', 'r', 's2', 'd', 'w')]
    wm = lambda x: (w * x).sum() / jnp.maximum(w.sum(), 1.0)

    def closs():
      q = self._q(self.q, s, a)
      noise = jnp.clip(
          jax.random.normal(nj.seed(), a.shape) * c.policy_noise,
          -c.noise_clip, c.noise_clip)
      a2 = jnp.clip(self._act(self.slowpol, s2) + noise, -1.0, 1.0)
      y = sg(r + c.discount * (1 - d) * self._q(self.slowq, s2, a2).min(0))
      return wm(((q - y) ** 2).sum(0)), {
          'q': wm(q.mean(0)), 'td': wm(jnp.abs(q - y).mean(0))}

    def ploss():
      pi = self._act(self.pol, s)
      q = self._q(self.q, s, pi).min(0)
      lam = c.alpha / (sg(wm(jnp.abs(q))) + 1e-6)
      bc = wm(((pi - b['raw']) ** 2).mean(-1))
      return -lam * wm(q) + bc, {'bc': bc, 'lambda': lam}

    def actor_step():
      mets, aux = self.opts['popt'](ploss, has_aux=True)
      self.slowpol.update()
      [x.update() for x in self.slowq]
      return {**mets, **aux}

    cm, caux = self.opts['copt'](closs, has_aux=True)
    pm = self._every(self.count.read() % c.policy_freq == 0, actor_step)
    return {**cm, **caux, **pm}

  def _iql(self, b):
    c = self.config
    s, a, r, s2, d, w = [b[k] for k in ('s', 'a', 'r', 's2', 'd', 'w')]
    wm = lambda x: (w * x).sum() / jnp.maximum(w.sum(), 1.0)
    value = lambda x: self.v(x)['v'][..., 0]

    def vloss():
      diff = sg(self._q(self.slowq, s, a).min(0)) - value(s)
      weight = jnp.where(diff > 0, c.expectile, 1 - c.expectile)
      return wm(weight * diff ** 2), {'v': wm(value(s))}

    def ploss():
      adv = sg(self._q(self.q, s, a).min(0) - value(s))
      weight = jnp.minimum(jnp.exp(c.beta * adv), c.adv_clip)
      logp = self._logp(s, b['raw'])
      return wm(-weight * logp), {
          'logp': wm(logp), 'adv': wm(adv), 'awr_weight': wm(weight)}

    def closs():
      q = self._q(self.q, s, a)
      y = sg(r + c.discount * (1 - d) * value(s2))
      return wm(((q - y) ** 2).sum(0)), {
          'q': wm(q.mean(0)), 'td': wm(jnp.abs(q - y).mean(0))}

    vm, vaux = self.opts['vopt'](vloss, has_aux=True)
    pm, paux = self.opts['popt'](ploss, has_aux=True)
    cm, caux = self.opts['copt'](closs, has_aux=True)
    [x.update() for x in self.slowq]
    return {**vm, **vaux, **pm, **paux, **cm, **caux}

  def _every(self, pred, fn):
    # Run fn, then keep every state change it made only where pred holds.
    ctx = nj.context()
    before = dict(ctx)
    out = fn()
    for key, value in list(ctx.items()):
      if key in before and value is not before[key]:
        ctx[key] = jnp.where(pred, value, before[key])
    return out

  def _norm(self, x):
    return self.on(x) if self.on else x

  def _act(self, net, s):
    return jnp.tanh(net(s)['a'])

  def _gauss(self, s):
    c = self.config
    out = self.pol(s)
    logstd = jnp.clip(out['logstd'], np.log(c.std_min), np.log(c.std_max))
    return jnp.clip(out['mean'], -c.mean_clip, c.mean_clip), logstd

  def _logp(self, s, a):
    mean, logstd = self._gauss(s)
    z = (a - mean) / jnp.exp(logstd)
    return (-0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - logstd).mean(-1)

  def _q(self, nets, s, a):
    x = jnp.concatenate([s, a], -1)
    return jnp.stack([net(x)['q'][..., 0] for net in nets])
