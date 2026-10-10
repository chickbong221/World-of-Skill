import h5py
import numpy as np

from embodied.run import env_eval
from offline_comp import tasks as tasks_mod
from offline_comp.dataset import OfflineCompDataset


def make_task(tmp_path, obj, obstacle, length=1000, seed=0):
  rng = np.random.default_rng(seed)
  path = tmp_path / f'Panda_{obj}_{obstacle}_PickPlace' / 'data.hdf5'
  path.parent.mkdir()
  timeouts = np.zeros(length, bool)
  timeouts[49::50] = True
  terminals = np.zeros(length, bool)
  terminals[[i for i in (30, 130, 777) if i < length]] = True
  with h5py.File(path, 'w') as f:
    obs = rng.normal(size=(length, 5)).astype(np.float32)
    obs[:, 0] = np.arange(length)  # identifies the row
    f['observations'] = obs
    f['actions'] = rng.normal(size=(length, 2)).astype(np.float32)
    f['rewards'] = rng.random(length).astype(np.float32)
    f['terminals'] = terminals
    f['timeouts'] = timeouts
  return tasks_mod.Task('Panda', obj, obstacle, 'PickPlace', str(path))


def reference_row(task, start, length):
  # Record t is stored row start + t. A row's end flag makes it the last record
  # of its episode and the next row a first record; a record's reward is the
  # previous row's, 0 on a first record.
  with h5py.File(task.path, 'r') as f:
    sl = slice(start, start + length)
    obs, act = f['observations'][sl], f['actions'][sl]
    rew, term, tout = f['rewards'][sl], f['terminals'][sl], f['timeouts'][sl]
  out = {k: [] for k in ('reward', 'is_first', 'is_last', 'is_terminal')}
  for t in range(length):
    first = t == 0 or bool(term[t - 1] or tout[t - 1])
    out['is_first'].append(first)
    out['is_last'].append(bool(term[t] or tout[t]))
    out['is_terminal'].append(bool(term[t]))
    out['reward'].append(0.0 if first else rew[t - 1])
  return dict(
      vector=obs, action=act, reward=np.array(out['reward'], np.float32),
      is_first=np.array(out['is_first']), is_last=np.array(out['is_last']),
      is_terminal=np.array(out['is_terminal']))


class TestDataset:

  def test_windows_match_reader(self, tmp_path):
    tasks = [make_task(tmp_path, 'Box', 'None', 1000, 1),
             make_task(tmp_path, 'Plate', 'GoalWall', 600, 2)]
    for length in (1, 2, 17):
      ds = OfflineCompDataset(tasks, length)
      index = np.array([0, 1, 1, 0, 1, 0])
      starts = np.array([0, 600 - length, 25, 1000 - length, 49 - length // 2,
                         130 - length // 2])
      out = ds._window(index, starts)
      for row, (i, start) in enumerate(zip(index, starts)):
        ref = reference_row(tasks[i], int(start), length)
        for key, value in ref.items():
          np.testing.assert_array_equal(out[key][row], value, err_msg=key)
        stepid = out['stepid'][row]
        assert all(int.from_bytes(bytes(s[:4]), 'big') == i for s in stepid)
        assert [int.from_bytes(bytes(s[12:]), 'big')
                for s in stepid] == list(range(length))
      assert out['vector'].dtype == np.float32
      assert out['task_axes'].shape == (6, length, 4)

  def test_episode_boundaries(self, tmp_path):
    task = make_task(tmp_path, 'Box', 'None', 200, 3)
    with h5py.File(task.path, 'r') as f:
      end = f['terminals'][:] | f['timeouts'][:]
      rew = f['rewards'][:]
    episode = np.r_[0, np.cumsum(end)[:-1]]  # episode id of every stored row
    starts = np.arange(0, 200 - 20 + 1)
    out = OfflineCompDataset([task], 20)._window(
        np.zeros(len(starts), int), starts)
    rows = out['vector'][..., 0].astype(int)
    assert (rows == starts[:, None] + np.arange(20)).all()
    same = episode[rows[:, 1:]] == episode[rows[:, :-1]]
    # A reset obs is always a first record, never glued to the previous episode.
    np.testing.assert_array_equal(out['is_first'][:, 1:], ~same)
    np.testing.assert_array_equal(out['is_last'], end[rows])
    assert not (out['is_last'][:, :-1] & same).any()
    expect = np.where(out['is_first'][:, 1:], 0.0, rew[rows[:, :-1]])
    np.testing.assert_array_equal(out['reward'][:, 1:], expect)
    assert (out['reward'][out['is_first']] == 0).all()

  def test_transition_pairs(self, tmp_path):
    # What the model-free baselines read: records (0, 1) with batch_length 1.
    task = make_task(tmp_path, 'Box', 'None', 200, 3)
    with h5py.File(task.path, 'r') as f:
      end = f['terminals'][:] | f['timeouts'][:]
      rew = f['rewards'][:]
    starts = np.arange(0, 199)
    out = OfflineCompDataset([task], 2)._window(
        np.zeros(len(starts), int), starts)
    valid = ~out['is_first'][:, 1]
    np.testing.assert_array_equal(valid, ~end[starts])  # only an end row is masked
    np.testing.assert_array_equal(out['reward'][valid, 1], rew[starts][valid])
    assert valid.sum() == 199 - end[:199].sum()

  def test_sample_stays_inside_tasks(self, tmp_path):
    tasks = [make_task(tmp_path, 'Box', 'None', 100, 1),
             make_task(tmp_path, 'Plate', 'GoalWall', 100, 2)]
    ds = OfflineCompDataset(tasks, 64, sampling='uniform_transition')
    for _ in range(50):
      batch = ds.sample(8)
      for row in range(8):
        task = tasks[batch['task_id'][row, 0]]
        with h5py.File(task.path, 'r') as f:
          obs = f['observations'][:]
        first = batch['vector'][row, 0]
        start = int(np.flatnonzero((obs == first).all(1))[0])
        assert start + 64 <= 100
        np.testing.assert_array_equal(batch['vector'][row], obs[start: start + 64])

  def test_sequential_schedule(self, tmp_path):
    tasks = [make_task(tmp_path, o, 'None', 100, i)
             for i, o in enumerate(('Box', 'Plate', 'Dumbbell'))]
    ds = OfflineCompDataset(
        tasks, 4, schedule='sequential', batches_per_task=2)
    ids = [int(ds.sample(3)['task_id'][0, 0]) for _ in range(7)]
    assert ids == [0, 0, 1, 1, 2, 2, 0]


class FakePool:
  """Env stand-in: a 7-step episode with known rewards and success flags."""

  horizon = 7

  def __init__(self, specs, horizon, workers):
    self.t = {key: 0 for key in specs}

  def _obs(self, key, reward=0.0, last=False, first=False):
    t = self.t[key]
    kind_a = key[0].endswith('A')  # A: success mid-episode, B: at the end
    success = (t in (2, 3)) if kind_a else (t >= self.horizon - 1)
    return {
        'vector': np.zeros(93, np.float32), 'reward': np.float32(reward),
        'is_first': first, 'is_last': last, 'is_terminal': False,
        'log/success': np.float32(success)}

  def step(self, actions):
    out = {}
    for key, action in actions.items():
      if action is None:
        self.t[key] = 0
        out[key] = self._obs(key, first=True)
      else:
        self.t[key] += 1
        out[key] = self._obs(
            key, reward=self._rate(key) * self.t[key],
            last=self.t[key] >= self.horizon)
    return out

  @staticmethod
  def _rate(key):
    return 1 + len(key[0]) % 3

  def close(self):
    pass


class FakeAgent:

  obs_space = {'vector': None, 'task_id': None, 'task_axes': None}

  def __init__(self):
    self.calls = []
    self.syncs = 0

  def sync_policy(self):
    self.syncs += 1

  def init_policy(self, batch_size):
    return ()

  def policy(self, carry, obs, mode='eval'):
    self.calls.append({k: v.shape for k, v in obs.items()})
    return carry, {'action': np.zeros((len(obs['vector']), 8))}, {}


class FakeTask:

  def __init__(self, name):
    self.name = name
    self.robot, self.obj = 'Panda', 'Box'
    self.obstacle, self.objective = 'None', 'PickPlace'


class TestEnvEval:

  def test_metrics_and_batching(self, monkeypatch):
    monkeypatch.setattr(env_eval, 'Pool', FakePool)
    horizon = FakePool.horizon
    train = [FakeTask('t1_A'), FakeTask('t22_B')]
    test = [FakeTask('t333_A')]
    agent = FakeAgent()
    out = env_eval.evaluate(
        agent, {'train': train, 'test': test}, 5, horizon, slots=2)
    for split, tasks in (('train', train), ('test', test)):
      metrics = out[split]
      returns, ends = [], []
      for task in tasks:
        rate = 1 + len(task.name) % 3
        ret = rate * horizon * (horizon + 1) / 2
        end = 0.0 if task.name.endswith('A') else 1.0
        assert abs(metrics[f'tasks/{task.name}/return'] - ret) < 1e-4
        assert metrics[f'tasks/{task.name}/success_once'] == 1.0
        assert metrics[f'tasks/{task.name}/success_at_end'] == end
        returns.append(ret)
        ends.append(end)
      assert abs(metrics['return'] - np.mean(returns)) < 1e-4
      assert metrics['success_at_end'] == np.mean(ends)
      assert metrics['tasks_evaluated'] == len(tasks)
    # The policy must see the latest trained weights before rolling out.
    assert agent.syncs == 1
    # 5 episodes over 2 slots: the longer slot runs 3 episodes in lockstep.
    assert len(agent.calls) == 3 * horizon
    assert all(call['vector'] == (6, 93) for call in agent.calls)
    assert all(call['task_axes'] == (6, 4) for call in agent.calls)
    assert 'log/success' not in agent.calls[0]

  def test_max_tasks(self, monkeypatch):
    monkeypatch.setattr(env_eval, 'Pool', FakePool)
    tasks = [FakeTask(f't{i}_A') for i in range(4)]
    out = env_eval.evaluate(
        FakeAgent(), {'train': tasks}, 2, FakePool.horizon, max_tasks=2,
        slots=1)
    assert out['train']['tasks_evaluated'] == 2
