import importlib.util
import io
import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).parents[2] / 'third_party' / 'tdmpc2' / 'tdmpc2'
sys.path.insert(0, str(ROOT))  # TD-MPC2 imports its modules as `common.*`
from common import MODEL_SIZE, math  # noqa: E402
from common.parser import cfg_to_dataclass  # noqa: E402

# Loaded by path: this package is also named tdmpc2.
_spec = importlib.util.spec_from_file_location('tdmpc2_official', ROOT / 'tdmpc2.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TDMPC2 = _mod.TDMPC2


def make_cfg(config, obs_dim, act_dim):
  """TD-MPC2's config.yaml with our overrides, parsed like its parse_cfg."""
  cfg = OmegaConf.load(ROOT / 'config.yaml')
  cfg.pop('defaults', None)
  cfg.update(
      task='composuite', obs='state', episodic=False, multitask=False,
      model_size=config.model_size, batch_size=config.batch_size,
      compile=config.compile, seed=config.seed, obs_shape={'state': [obs_dim]},
      action_dim=act_dim, episode_length=config.episode_length,
      tasks=['composuite'], task_dim=0, task_title='CompoSuite')
  for k, v in MODEL_SIZE[config.model_size].items():
    cfg[k] = v
  cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
  return cfg_to_dataclass(cfg)


class Agent:
  """TD-MPC2 (Hansen et al., ICLR 2024) on the offline windows: its own update
  and MPPI planner, single-task style on the pooled train tasks (the task is in
  the observation). Windows are horizon + 1 steps; ones crossing an episode
  boundary are dropped, so the data batch carries a small margin."""

  def __init__(self, obs_space, act_space, config):
    self.obs_space, self.act_space, self.config = obs_space, act_space, config
    torch.manual_seed(config.seed)
    self.adim = act_space['action'].shape[0]
    self.cfg = make_cfg(config, obs_space['vector'].shape[0], self.adim)
    self.agent = TDMPC2(self.cfg)
    self.device = self.agent.device
    self.H, self.B = self.cfg.horizon, self.cfg.batch_size

  def stream(self, st):
    H, B = self.H, self.B

    def fn(data):
      # Record t holds obs_t, a_t and r_{t-1}; a window is valid when no record
      # after the first starts an episode.
      ok = ~data['is_first'][:, 1:H + 1].any(1)
      idx = np.flatnonzero(ok)
      idx = idx[np.arange(B) % len(idx)]
      win = {
          'obs': data['vector'][idx, :H + 1],
          'action': np.clip(data['action'][idx, :H], -1, 1),
          'reward': data['reward'][idx, 1:H + 1, None],
          'term': data['is_terminal'][idx, :H, None].astype(np.float32)}
      win = {k: torch.as_tensor(np.ascontiguousarray(v.swapaxes(0, 1)),
                                dtype=torch.float32).to(self.device)
             for k, v in win.items()}
      win['valid_frac'] = float(ok.mean())
      return win
    import embodied
    return embodied.streams.Prefetch(st, fn)

  def init_train(self, batch_size):
    return None

  def init_report(self, batch_size):
    return None

  def init_policy(self, batch_size):
    return torch.zeros(self.H, batch_size, self.adim, device=self.device)

  def sync_policy(self):
    pass

  def train(self, carry, data):
    torch.compiler.cudagraph_mark_step_begin()
    info = self.agent._update(
        data['obs'], data['action'], data['reward'], data['term'])
    self.steps = getattr(self, 'steps', 0) + 1
    mets = {}
    if self.steps % 100 == 0:  # avoid a device sync on every update
      mets = {k: float(v) for k, v in info.items()}
      mets['valid_frac'] = data['valid_frac']
    return carry, {}, mets

  def report(self, carry, data):
    return carry, {}

  @torch.no_grad()
  def policy(self, carry, obs, mode='train'):
    obs_ = torch.as_tensor(obs['vector'], device=self.device).float()
    t0 = torch.as_tensor(obs['is_first'], device=self.device)
    action, carry = self._plan(obs_, carry, t0, eval_mode=(mode == 'eval'))
    return carry, {'action': action.cpu().numpy()}, {}

  @torch.no_grad()
  def _plan(self, obs, prev_mean, t0, eval_mode):
    # TDMPC2._plan (MPPI) with a leading batch of N environments.
    cfg, model = self.cfg, self.agent.model
    N, H, A = obs.shape[0], cfg.horizon, cfg.action_dim
    S, P, K = cfg.num_samples, cfg.num_pi_trajs, cfg.num_elites
    z = model.encode(obs, None)
    actions = torch.empty(H, N, S, A, device=self.device)
    if P > 0:
      _z = z.repeat_interleave(P, 0)
      for t in range(H):
        a, _ = model.pi(_z, None)
        actions[t, :, :P] = a.view(N, P, A)
        if t < H - 1:
          _z = model.next(_z, a, None)
    z = z.repeat_interleave(S, 0)
    mean = torch.zeros(H, N, A, device=self.device)
    mean[:-1] = torch.where(t0[None, :, None], 0.0, prev_mean[1:])
    std = torch.full((H, N, A), cfg.max_std, device=self.device)
    for _ in range(cfg.iterations):
      r = torch.randn(H, N, S - P, A, device=self.device)
      actions[:, :, P:] = (mean[:, :, None] + std[:, :, None] * r).clamp(-1, 1)
      value = self._value(z, actions.view(H, N * S, A)).nan_to_num(0)
      value = value.view(N, S)
      elite = torch.topk(value, K, dim=1).indices
      elite_value = torch.gather(value, 1, elite)
      elite_actions = torch.gather(
          actions, 2, elite[None, :, :, None].expand(H, N, K, A))
      score = torch.exp(cfg.temperature * (
          elite_value - elite_value.max(1, keepdim=True).values))
      score = score / score.sum(1, keepdim=True)
      w = score[None, :, :, None]
      mean = (w * elite_actions).sum(2) / (score.sum(1)[None, :, None] + 1e-9)
      std = ((w * (elite_actions - mean[:, :, None]) ** 2).sum(2) / (
          score.sum(1)[None, :, None] + 1e-9)).sqrt()
      std = std.clamp(cfg.min_std, cfg.max_std)
    gumbel = -torch.empty_like(score).exponential_().log()
    pick = (score.log() + gumbel).argmax(1)
    a = elite_actions[0, torch.arange(N, device=self.device), pick]
    if not eval_mode:
      a = a + std[0] * torch.randn(N, A, device=self.device)
    return a.clamp(-1, 1), mean

  @torch.no_grad()
  def _value(self, z, actions):
    # TDMPC2._estimate_value; CompoSuite has no terminations.
    model, G, discount = self.agent.model, 0, 1
    for t in range(self.cfg.horizon):
      reward = math.two_hot_inv(model.reward(z, actions[t], None), self.cfg)
      z = model.next(z, actions[t], None)
      G = G + discount * reward
      discount = discount * self.agent.discount
    action, _ = model.pi(z, None)
    return (G + discount * model.Q(z, action, None, return_type='avg'))[:, 0]

  def save(self):
    buf = io.BytesIO()
    torch.save({
        'model': self.agent.model.state_dict(),
        'optim': self.agent.optim.state_dict(),
        'pi_optim': self.agent.pi_optim.state_dict(),
        'scale': self.agent.scale.state_dict(),
        'steps': getattr(self, 'steps', 0)}, buf)
    return buf.getvalue()

  def load(self, data):
    state = torch.load(io.BytesIO(data), map_location=self.device,
                       weights_only=False)
    self.agent.model.load_state_dict(state['model'])
    self.agent.optim.load_state_dict(state['optim'])
    self.agent.pi_optim.load_state_dict(state['pi_optim'])
    self.agent.scale.load_state_dict(state['scale'])
    self.steps = state['steps']
