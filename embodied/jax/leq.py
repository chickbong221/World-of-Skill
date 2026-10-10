"""LEQ (Park & Lee, ICLR 2025) on a frozen latent world model.

Policy phase of the two_phase schedule when agent.ac == 'leq', following
kwanyoungpark/LEQ (DPG_lambda_update_actor, lambda_update_q): a deterministic
actor trained DDPG-style by expectile-weighted gradients of lambda-returns
through the model, and a Q critic trained by lower-expectile regression of the
imagined lambda-returns plus TD on the data, with an EMA regulariser.
"""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import nets
from .heads import MLPHead
from .opt import Optimizer
from .utils import SlowModel

f32 = jnp.float32
sg = jax.lax.stop_gradient
flat = lambda x: x.reshape((-1, *x.shape[2:]))
stack = lambda xs: jax.tree.map(lambda *x: jnp.stack(x), *xs)


def setup(agent, config):
  c = config.leq
  scalar = elements.Space(np.float32, ())
  agent.q = MLPHead(scalar, **c.critic, name='q')
  agent.slowq = SlowModel(
      MLPHead(scalar, **c.critic, name='slowq'),
      source=agent.q, rate=c.target_rate)
  lr = c.actor_lr
  if c.anneal > 0:  # LEQ decays the actor lr to 0 over the policy updates
    lr = optax.cosine_decay_schedule(c.actor_lr, c.anneal)
  agent.opt_pi = Optimizer([agent.pol], optax.adam(lr), name='opt_pi')
  agent.opt_q = Optimizer([agent.q], optax.adam(c.critic_lr), name='opt_q')


def act(agent, feat):
  pol = agent.pol(agent.feat2tensor(feat), 1)
  return {k: v.pred() for k, v in pol.items()}


def qval(head, agent, feat, action):
  x = [agent.feat2tensor(feat)] + [nets.cast(action[k]) for k in agent.act_space]
  return head(jnp.concatenate(x, -1), 1).pred()


def rollout(agent, start, H, delta):
  carry, feats, acts = start, [start], []
  for t in range(H + 1):
    a = act(agent, carry)
    a = {k: v + delta[k][t] for k, v in a.items()}
    acts.append(a)
    if t < H:
      carry, _ = agent.dyn.imagine(carry, a, 1, False, single=True)
      feats.append(carry)
  return stack(feats), stack(acts)


def lambda_return(q, rew, live, mask, lam):
  # LEQ's truncated lambda-return, normalised over the available n-step
  # returns. coef[t] = d ret[t] / d ret[t + 1].
  rets, coefs, c = [q[-1]], [], 1.0
  for t in reversed(range(len(rew))):
    coefs.append(lam * c * mask[t] * live[t] / (1 + lam * c))
    rets.append((q[t] + lam * c * mask[t] * (rew[t] + live[t] * rets[-1]))
                / (1 + lam * c))
    c = 1 + lam * c
  return jnp.stack(rets[::-1]), jnp.stack(coefs[::-1])


def train(agent, carry, data):
  c = agent.config.leq
  H = agent.config.imag_length
  disc = 1 - 1 / agent.config.horizon
  carry, obs, prevact, dataact, _ = agent._apply_replay_context(carry, data)
  enc_carry, dyn_carry, dec_carry = carry
  reset = obs['is_first']
  B, T = reset.shape
  enc_carry, _, tokens = agent.enc(enc_carry, obs, reset, False)
  dyn_carry, _, feat = agent.dyn.observe(
      dyn_carry, tokens, prevact, reset, False)
  feat = sg({k: feat[k] for k in ('deter', 'stoch')})
  start = jax.tree.map(flat, feat)
  N = B * T

  # Imagined rollouts from every posterior state, differentiated w.r.t.
  # additive action offsets: d ret[0] / d a[t] = prod(coef[:t]) d ret[t] / d a[t]
  # since a[t] reaches ret[0] only through ret[t] (masks are constants).
  def objective(delta):
    feats, acts = rollout(agent, start, H, delta)
    q = qval(agent.q, agent, jax.tree.map(flat, feats),
             jax.tree.map(flat, acts)).reshape((H + 1, N))
    nxt = agent.feat2tensor(jax.tree.map(lambda x: flat(x[1:]), feats))
    rew = agent.rew(nxt, 1).pred().reshape((H, N))
    live = sg(agent.con(nxt, 1).prob(1).reshape((H, N)))
    live = live if agent.config.contdisc else disc * live
    mask = jnp.cumprod(jnp.concatenate([jnp.ones((1, N)), live / disc]), 0)
    ret, coef = lambda_return(q, rew, live, mask, c.lam)
    return ret[0].sum(), (feats, acts, q, rew, live, ret, coef)

  zeros = {k: jnp.zeros((H + 1, N, *v.shape), f32)
           for k, v in agent.act_space.items()}
  _, _, grads, aux = nj.grad(objective, 0, has_aux=True)(zeros)
  feats, acts, q, rew, live, ret, coef = sg(aux)
  prefix = jnp.cumprod(jnp.concatenate([jnp.ones((1, N)), coef]), 0)
  ok = prefix > 1e-8
  dirs = {k: jnp.where(ok, 1 / jnp.where(ok, prefix, 1), 0)[..., None] * v
          for k, v in grads.items()}
  weight = jnp.where(ret > q, c.expectile, 1 - c.expectile).at[H].set(0.5)

  def actor_loss(feats, dirs, weight):
    a = act(agent, jax.tree.map(flat, feats))
    obj = sum((dirs[k] * a[k].reshape(dirs[k].shape)).sum(-1) for k in a)
    return -(weight * obj).mean(1).sum(), {}

  # Critic: the same rollouts (old actor and critic, as in LEQ) and the data.
  lw = jnp.cumprod(jnp.concatenate([jnp.ones((1, N)), live[:-1]]), 0)
  img = jax.tree.map(lambda x: flat(x[:H]), (feats, acts))
  real = jax.tree.map(lambda x: flat(x[:, :-1]), feat)
  nxt = jax.tree.map(lambda x: flat(x[:, 1:]), feat)
  reala = {k: jnp.clip(flat(v[:, :-1]), -1, 1) for k, v in dataact.items()}
  valid = f32(~flat(obs['is_first'][:, 1:]))
  live_data = disc * f32(~flat(obs['is_terminal'][:, :-1]))  # row t ends t
  target = flat(obs['reward'][:, 1:]) + live_data * qval(
      agent.q, agent, nxt, act(agent, nxt))
  target, tar_img = sg((target, ret[:H].reshape(-1)))
  slow_img = sg(qval(agent.slowq, agent, *img))
  slow_data = sg(qval(agent.slowq, agent, real, reala))
  lw, beta = lw.reshape(-1), c.ratio
  vmean = lambda x: (x * valid).sum() / jnp.maximum(valid.sum(), 1)

  def critic_loss():
    q_img = qval(agent.q, agent, *img)
    q_data = qval(agent.q, agent, real, reala)
    diff = tar_img - q_img
    loss_img = (jnp.where(diff > 0, c.expectile, 1 - c.expectile)
                * jnp.square(diff) * lw).mean()
    loss_data = vmean(0.5 * jnp.square(target - q_data))
    reg = (beta * (jnp.square(slow_img - q_img) * lw).mean()
           + (1 - beta) * vmean(jnp.square(slow_data - q_data)))
    loss = (1 - beta) * loss_data + beta * loss_img + reg
    mets = dict(loss_img=loss_img, loss_data=loss_data, reg=reg,
                q_img=q_img.mean(), q_data=vmean(q_data))
    return loss, mets

  metrics, _ = agent.opt_pi(actor_loss, feats, dirs, weight, has_aux=True)
  mets, cmets = agent.opt_q(critic_loss, has_aux=True)
  agent.slowq.update()
  metrics.update(mets)
  metrics.update({f'leq/{k}': v for k, v in cmets.items()})
  metrics.update({
      'leq/ret': ret[0].mean(), 'leq/q': q.mean(), 'leq/rew': rew.mean(),
      'leq/live': live.mean(), 'leq/above': f32(ret[:H] > q[:H]).mean(),
      'leq/dir_rms': jnp.sqrt(sum(jnp.square(v).mean() for v in dirs.values())),
      'leq/rew_data': vmean(flat(obs['reward'][:, 1:]))})
  carry = (enc_carry, dyn_carry, dec_carry,
           {k: data[k][:, -1] for k in agent.act_space})
  return carry, {}, metrics
