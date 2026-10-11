import io
import pathlib
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).parents[2] / 'third_party' / 'c-lap'
sys.path.insert(0, str(ROOT))
import rlab_jax  # noqa: E402,F401  (registers the C-LAP modules)
from rlab_jax.modules import Agent as ClapAgent, SequenceModel  # noqa: E402
from rlab_jax.rlab_dataclasses import Transition  # noqa: E402
from rlab_jax.utils.cfg import load_cfg  # noqa: E402
from rlab_jax.utils.keygen import keygen  # noqa: E402
from rlab_jax.utils.rollout import DreamRollout  # noqa: E402
from rlab_jax_projects.latent_actions.modules import MeanStdScaler  # noqa: E402

f32 = jnp.float32


class Agent:
  """C-LAP (Alles et al., NeurIPS 2024) with its own latent-action RSSM, agent
  and training steps (c-lap.yaml, the D4RL state-observation setting), fed
  with offline CompoSuite windows: the world model phase runs its sequence
  model step, the policy phase its agent step on the frozen model. Windows of
  window + 1 records that cross an episode boundary are dropped, so the data
  batch carries a margin."""

  def __init__(self, obs_space, act_space, config):
    self.obs_space, self.act_space, self.config = obs_space, act_space, config
    keygen.set(config.seed)
    cfg = load_cfg(config.cfg)
    cfg.agent.latent_action_std_scaling = config.latent_action_std_scaling
    self.cfg, self.B, self.T = cfg, cfg.experiment.batch_size, config.window
    odim, adim = obs_space['vector'].shape[0], act_space['action'].shape[0]
    self.seqm = SequenceModel.create(
        cfg=cfg.seqm, n_observations=odim, n_actions=adim, env_name='composuite')
    self.ac = ClapAgent.create(
        cfg=cfg.agent, n_inputs=self.seqm.n_latent,
        n_actions=cfg.seqm.latent_action_model.n_latent_actions,
        decode_fn=self.seqm.latent_action_model._decode)
    self.ac.dream_generate_fn = DreamRollout(
        sequence_model=self.seqm, act_fn=self.ac.dream_act,
        n_rollouts=cfg.experiment.dream_rollout_batch_size,
        steps_per_rollout=cfg.experiment.dream_rollout_steps).generate
    self.model_state = self.seqm.initial_state()
    self.agent_state = self.ac.initial_state()
    self.key = jax.random.PRNGKey(config.seed)
    self.steps = 0

    seqm, ac = self.seqm, self.ac

    @eqx.filter_jit
    def wm_step(model_state, batch, key):
      model_state, _, log = seqm.training_step(
          model_state=model_state, batch=batch, key=key)
      return model_state, log

    @eqx.filter_jit
    def ac_step(model_state, agent_state, batch, key):
      k1, k2 = jax.random.split(key)
      preds = seqm(
          model=model_state.module,
          observation_trajectory=batch.observation_trajectory,
          action_trajectory=batch.action_trajectory, key=k1,
          scaler=model_state.scaler)
      return ac.training_step(
          model_state=model_state, agent_state=agent_state, batch=batch,
          predictions=preds, key=k2)

    # As C-LAP's BatchRollout: the first action of an episode comes from the
    # initial latent state, later ones from the state filtered with the new
    # observation and the previous (latent) action.
    @eqx.filter_jit
    def act(model_state, agent_state, obs, latent, action, latent_action, key):
      k1, k2 = jax.random.split(key)
      if latent is None:
        latent = seqm.initial_latent_state(model_state.module, obs.shape[0])
      else:
        latent = seqm.filter(
            model_state=model_state, next_observation=obs, latent_state=latent,
            latent_action=latent_action, action=action, key=k1)
      action, latent_action = ac.evaluate(
          latent_state=latent, model_state=model_state, key=k2,
          agent_state=agent_state, deterministic=True)
      return latent, action, latent_action

    self._wm_step, self._ac_step, self._act = wm_step, ac_step, act

  def fit_data(self, dataset):
    # C-LAP standardises observations with the dataset mean and std; constant
    # dimensions (here the fixed robot / objective indicators) keep std 1.
    mean, std = dataset.obs_stats()
    std = np.where(std < 1e-6, 1.0, std)
    self.model_state.scaler = MeanStdScaler(
        mean=jnp.asarray(mean), std=jnp.asarray(std))

  def stream(self, st):
    B, T = self.B, self.T

    def fn(data):
      # Record t holds obs_t, a_t and r_{t-1}; a window is valid when no record
      # after the first starts an episode.
      ok = ~data['is_first'][:, 1:T + 1].any(1)
      idx = np.flatnonzero(ok)
      idx = idx[np.arange(B) % len(idx)]
      tm = lambda x: jnp.asarray(np.ascontiguousarray(x.swapaxes(0, 1)), f32)
      vec = data['vector'][idx]
      return Transition(
          observation=tm(vec[:, :T]), next_observation=tm(vec[:, 1:T + 1]),
          action=tm(np.clip(data['action'][idx, :T], -1, 1)),
          reward=tm(data['reward'][idx, 1:T + 1, None]),
          done=tm(data['is_terminal'][idx, :T, None]),
          mask=jnp.ones((T, B, 1), f32))
    import embodied
    return embodied.streams.Prefetch(st, fn)

  def init_train(self, batch_size):
    return None

  def init_report(self, batch_size):
    return None

  def init_policy(self, batch_size):
    return (None, None, None)

  def sync_policy(self):
    pass

  def train(self, carry, data, phase='policy'):
    self.key, key = jax.random.split(self.key)
    if phase == 'wm':
      self.model_state, log = self._wm_step(self.model_state, data, key)
    else:
      self.agent_state, log = self._ac_step(
          self.model_state, self.agent_state, data, key)
    self.steps += 1
    mets = {}
    if self.steps % 100 == 0:  # avoid a device sync on every update
      mets = {f'{phase}/{k}': float(np.mean(v)) for k, v in log.items()}
    return carry, {}, mets

  def report(self, carry, data):
    return carry, {}

  def policy(self, carry, obs, mode='train'):
    self.key, key = jax.random.split(self.key)
    first = np.asarray(obs['is_first'])
    if first.all():
      carry = (None, None, None)
    # env_eval steps all envs in lockstep with one horizon, so they reset
    # together.
    assert not first.any() or first.all(), 'per-env resets are not supported'
    latent, action, latent_action = self._act(
        self.model_state, self.agent_state, jnp.asarray(obs['vector'], f32),
        *carry, key)
    return (latent, action, latent_action), {
        'action': np.asarray(action.sample)}, {}

  def save(self):
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(
        buf, (self.model_state, self.agent_state, self.key))
    return {'state': buf.getvalue(), 'steps': self.steps}

  def load(self, data):
    self.model_state, self.agent_state, self.key = eqx.tree_deserialise_leaves(
        io.BytesIO(data['state']),
        (self.model_state, self.agent_state, self.key))
    self.steps = data['steps']
