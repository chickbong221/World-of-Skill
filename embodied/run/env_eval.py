import contextlib
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np


def _step(envs, requests):
  return {k: envs[k].step({
      'action': np.zeros(envs[k].action_dim, np.float32) if a is None else a,
      'reset': a is None}) for k, a in requests.items()}


def _worker(conn):
  try:
    from embodied.envs.offline_comp import CompoSuiteEval
    envs = {}
    while True:
      cmd, arg = conn.recv()
      if cmd == 'seed':
        np.random.seed(arg)
        conn.send(None)
      elif cmd == 'make':
        for key, (spec, horizon) in arg.items():
          envs[key] = CompoSuiteEval(*spec, horizon=horizon)
        conn.send(None)
      elif cmd == 'step':
        conn.send(_step(envs, arg))
      else:
        break
  except BaseException:
    conn.send(traceback.format_exc())
  finally:
    conn.close()


@contextlib.contextmanager
def _hide_main():
  # Spawned workers would otherwise re-import (and re-run) the training script.
  main = sys.modules['__main__']
  saved = {k: main.__dict__.pop(k) for k in ('__file__', '__spec__')
           if k in main.__dict__}
  main.__spec__ = None
  try:
    yield
  finally:
    main.__dict__.update(saved)


class Pool:
  """CompoSuite envs spread over worker processes, stepped in lockstep."""

  def __init__(self, specs, horizon, workers, seed=None, timeout=600):
    ctx = mp.get_context('spawn')
    keys = list(specs)
    n = max(1, min(workers, len(keys)))
    self.owner = {k: i % n for i, k in enumerate(keys)}
    self.timeout = timeout
    self.conns, self.procs = [], []
    try:
      for _ in range(n):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker, args=(child,), daemon=True)
        with _hide_main():
          proc.start()
        child.close()
        self.conns.append(parent)
        self.procs.append(proc)
      for i, conn in enumerate(self.conns):
        if seed is not None:
          conn.send(('seed', seed + i))
          self._recv(conn)
        conn.send(('make', {
            k: (specs[k], horizon) for k in keys if self.owner[k] == i}))
      [self._recv(conn) for conn in self.conns]
    except BaseException:
      self.close()
      raise

  def step(self, actions):
    """actions: {key: action, or None to reset} -> {key: obs}."""
    reqs = [{} for _ in self.conns]
    for key, act in actions.items():
      reqs[self.owner[key]][key] = act
    for conn, req in zip(self.conns, reqs):
      req and conn.send(('step', req))
    out = {}
    for conn, req in zip(self.conns, reqs):
      req and out.update(self._recv(conn))
    return out

  def close(self):
    for conn in self.conns:
      try:
        conn.send(('close', None))
      except (OSError, ValueError):
        pass
    for proc in self.procs:
      proc.join(5)
      proc.is_alive() and proc.terminate()
    [conn.close() for conn in self.conns]

  def _recv(self, conn):
    if not conn.poll(self.timeout):
      raise TimeoutError(f'env worker silent for {self.timeout}s')
    out = conn.recv()
    if isinstance(out, str):
      raise RuntimeError(f'env worker failed:\n{out}')
    return out


class SerialPool:
  """Same interface as Pool but steps every env in this process."""

  def __init__(self, specs, horizon, seed=None):
    from embodied.envs.offline_comp import CompoSuiteEval
    seed is None or np.random.seed(seed)
    self.envs = {k: CompoSuiteEval(*s, horizon=horizon)
                 for k, s in specs.items()}

  def step(self, actions):
    return _step(self.envs, actions)

  def close(self):
    [env.close() for env in self.envs.values()]


def evaluate(
    agent, splits, episodes, horizon, max_tasks=0, slots=1, workers=0):
  """Roll out `agent.policy` in live CompoSuite envs, batched over envs.

  `splits` maps a name to its tasks. Every task gets `slots` envs that share
  its `episodes` episodes, and all envs share one batched policy call per
  step. `workers` is the number of env processes (0: half the cpus, <0: step
  the envs in this process). Returns {split: metrics} with return /
  success_once / success_at_end averaged over the split and per task.
  """
  from offline_comp import tasks as tasks_mod

  slots = max(1, min(int(slots), int(episodes)))
  envs, quota = [], {}  # envs: [(split, task, key)]
  for name, tasks in splits.items():
    tasks = list(tasks)[:max_tasks] if max_tasks else list(tasks)
    for task in tasks:
      for slot in range(slots):
        key = (task.name, slot)
        envs.append((name, task, key))
        quota[key] = episodes // slots + int(slot < episodes % slots)
  keys = [key for _, _, key in envs]
  specs = {key: (t.robot, t.obj, t.obstacle, t.objective)
           for _, t, key in envs}
  extra = {}
  if 'task_id' in getattr(agent, 'obs_space', {}):
    extra = {'task_id': np.zeros(len(keys), np.int32), 'task_axes': np.array([[
        tasks_mod.ROBOTS.index(t.robot), tasks_mod.OBJECTS.index(t.obj),
        tasks_mod.OBSTACLES.index(t.obstacle),
        tasks_mod.OBJECTIVES.index(t.objective)] for _, t, _ in envs],
        np.int32)}

  getattr(agent, 'sync_policy', lambda: None)()
  if workers < 0:
    pool = SerialPool(specs, horizon)
  else:
    pool = Pool(specs, horizon, workers or max(1, (os.cpu_count() or 2) // 2))
  try:
    carry = agent.init_policy(len(keys))
    obs = pool.step({key: None for key in keys})
    ret = dict.fromkeys(keys, 0.0)
    once = dict.fromkeys(keys, False)
    results = {key: ([], [], []) for key in keys}  # return, once, at_end
    active = set(keys)
    while active:
      for key in active:
        once[key] |= float(obs[key]['log/success']) > 0.5
      batch = {
          k: np.stack([obs[key][k] for key in keys])
          for k in obs[keys[0]] if not k.startswith('log/')}
      batch.update(extra)
      carry, acts, _ = agent.policy(carry, batch, mode='eval')
      action = np.asarray(acts['action'])
      new = pool.step({
          key: action[i] for i, key in enumerate(keys) if key in active})
      resets = {}
      for key, o in new.items():
        obs[key] = o
        ret[key] += float(o['reward'])
        if not bool(o['is_last']):
          continue
        success = float(o['log/success'])
        rets, onces, ends = results[key]
        rets.append(ret[key])
        onces.append(float(once[key] or success > 0.5))
        ends.append(success)
        ret[key], once[key] = 0.0, False
        if len(rets) < quota[key]:
          resets[key] = None
        else:
          active.discard(key)
      if resets:
        obs.update(pool.step(resets))
  finally:
    pool.close()

  out = {}
  for name in splits:
    per_task = {}
    for split, task, key in envs:
      if split == name:
        lists = per_task.setdefault(task.name, ([], [], []))
        [dst.extend(src) for dst, src in zip(lists, results[key])]
    flat = [sum((v[i] for v in per_task.values()), []) for i in range(3)]
    metrics = {}
    if flat[0]:
      metrics = {
          'return': float(np.mean(flat[0])),
          'success_once': float(np.mean(flat[1])),
          'success_at_end': float(np.mean(flat[2])),
          'tasks_evaluated': float(len(per_task))}
    for task_name, values in per_task.items():
      for label, value in zip(('return', 'success_once', 'success_at_end'),
                              values):
        metrics[f'tasks/{task_name}/{label}'] = float(np.mean(value))
    out[name] = metrics
  return out
