import itertools
import pathlib
import time

import elements
import numpy as np

from . import selection
from . import tasks as tasks_mod


KEYS = ("observations", "actions", "rewards", "terminals", "timeouts")


class Store:
  """All selected task files decompressed once into flat RAM arrays.

  Random windows in the (gzip, column-chunked) HDF5 files cost ~20 ms per
  window; gathering from RAM costs microseconds.
  """

  def __init__(self, tasks):
    import h5py
    tasks = list({t.path: t for t in tasks}.values())
    lengths, shapes = [], []
    for task in tasks:
      with h5py.File(task.path, "r") as f:
        missing = [k for k in KEYS if k not in f]
        if missing:
          raise ValueError(f"{task.path} is missing HDF5 keys: {missing}")
        lengths.append(int(f["observations"].shape[0]))
        shapes.append((f["observations"].shape[1:], f["actions"].shape[1:]))
    if len(set(shapes)) > 1:
      raise ValueError(f"Incompatible obs/action shapes across tasks: {set(shapes)}")
    self.obs_shape, self.action_shape = shapes[0]
    total = sum(lengths)
    self.obs = np.empty((total, *self.obs_shape), np.float32)
    self.action = np.empty((total, *self.action_shape), np.float32)
    self.reward = np.empty((total,), np.float32)
    self.terminal = np.empty((total,), bool)
    self.timeout = np.empty((total,), bool)
    dsts = (self.obs, self.action, self.reward, self.terminal, self.timeout)
    self.span = {}  # path -> (start, length) into the flat arrays
    start, t0 = 0, time.time()
    for task, n in zip(tasks, lengths):
      self.span[task.path] = (start, n)
      with h5py.File(task.path, "r") as f:
        for key, dst in zip(KEYS, dsts):
          f[key].read_direct(dst, dest_sel=np.s_[start: start + n])
      start += n
    print(f"Loaded {len(tasks)} tasks ({total / 1e6:.1f}M transitions, "
          f"{sum(x.nbytes for x in dsts) / 2 ** 30:.1f} GiB) in "
          f"{time.time() - t0:.0f}s")


class OfflineCompDataset:

  def __init__(
      self, tasks, sequence_length, seed=0, sampling="uniform_task",
      schedule="mixed", batches_per_task=1000, shuffle_tasks=False,
      store=None):
    self.tasks = list(tasks)
    self.sequence_length = int(sequence_length)
    self.sampling = sampling
    self.schedule = schedule
    self.batches_per_task = int(batches_per_task)
    self.shuffle_tasks = _as_bool(shuffle_tasks)
    self.rng = np.random.default_rng(seed)
    self._counter = itertools.count()
    self._task_position = 0
    self._task_batch_count = 0
    self._task_order = []

    self.store = store or Store(self.tasks)
    self.obs_shape = self.store.obs_shape
    self.action_shape = self.store.action_shape
    spans = [self.store.span[t.path] for t in self.tasks]
    self.base = np.array([s for s, _ in spans], np.int64)
    self.lengths = np.array([n for _, n in spans], np.int64)
    if np.any(self.lengths < self.sequence_length):
      bad = [t.name for t, n in zip(self.tasks, self.lengths)
             if n < self.sequence_length]
      raise ValueError(f"Tasks shorter than sequence length: {bad}")
    if self.schedule not in ("mixed", "sequential"):
      raise ValueError(
          "sampling.schedule must be 'mixed' or 'sequential', "
          f"got {self.schedule!r}")
    if self.batches_per_task < 1:
      raise ValueError("sampling.batches_per_task must be at least 1")
    if self.sampling not in ("uniform_task", "uniform_transition"):
      raise ValueError(
          "sampling.mode must be 'uniform_task' or 'uniform_transition', "
          f"got {self.sampling!r}")
    weights = self.lengths.astype(np.float64)
    self.transition_probs = weights / weights.sum()
    self.axes = np.array([[
        tasks_mod.ROBOTS.index(t.robot),
        tasks_mod.OBJECTS.index(t.obj),
        tasks_mod.OBSTACLES.index(t.obstacle),
        tasks_mod.OBJECTIVES.index(t.objective),
    ] for t in self.tasks], np.int32)
    self._steps = np.arange(self.sequence_length)
    self._reset_task_order()

  @classmethod
  def from_config(cls, data_config, split, sequence_length, seed=0):
    root = _get(data_config, "root")
    if not root:
      raise ValueError("data.root must point at an extracted Dryad archive")
    _, train_tasks, test_tasks = selection.resolve_selection(
        root, _get(data_config, "train"), _get(data_config, "test"),
        _as_bool(_get(data_config, "allow_overlap", False)))
    selected = train_tasks if split == "train" else test_tasks
    sampling_config = _get(data_config, "sampling", {})
    schedule = _get(sampling_config, "schedule", "mixed")
    if split != "train":
      schedule = _get(sampling_config, "eval_schedule", "mixed")
    return cls(
        selected, sequence_length, seed=seed,
        sampling=_get(sampling_config, "mode", "uniform_task"),
        schedule=schedule,
        batches_per_task=_get(sampling_config, "batches_per_task", 1000),
        shuffle_tasks=_get(sampling_config, "shuffle_tasks", False))

  def close(self):
    pass

  @property
  def obs_space(self):
    return {
        "vector": elements.Space(np.float32, self.obs_shape),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
        # MoSS: environment index for per-environment responsibility, plus
        # the CompoSuite (robot, object, obstacle, objective) component ids.
        "task_id": elements.Space(np.int32, (), 0, max(len(self.tasks), 1)),
        "task_axes": elements.Space(np.int32, (4,), 0, 4),
    }

  @property
  def act_space(self):
    return {
        "action": elements.Space(np.float32, self.action_shape, -1.0, 1.0),
    }

  def sample(self, batch_size):
    if self.schedule == "mixed":
      tasks = self._choose_tasks(batch_size)
    else:
      tasks = np.full(batch_size, self._choose_sequential_task())
    starts = self.rng.integers(0, self.lengths[tasks] - self.sequence_length + 1)
    return self._window(tasks, starts)

  def _window(self, tasks, starts):
    batch_size, T = len(tasks), self.sequence_length
    idx = (self.base[tasks] + starts)[:, None] + self._steps
    store = self.store
    end = store.terminal[idx] | store.timeout[idx]

    # A stored row is (obs_t, a_t, r_t, end_t) and the obs after an episode's
    # last row is not stored, so that row is the last record of its episode and
    # the next row (the new episode's reset obs) is a first record. The reward
    # of a record is the previous row's, 0 after a reset.
    is_last = end
    is_terminal = store.terminal[idx]
    is_first = np.zeros((batch_size, T), bool)
    is_first[:, 0] = True
    is_first[:, 1:] = end[:, :-1]
    reward = np.zeros((batch_size, T), np.float32)
    reward[:, 1:] = np.where(is_first[:, 1:], 0.0, store.reward[idx[:, :-1]])

    counters = np.fromiter(
        itertools.islice(self._counter, batch_size), np.int64, batch_size)
    be = lambda x: np.ascontiguousarray(x.astype(x.dtype.newbyteorder(">")))
    stepid = np.empty((batch_size, T, 20), np.uint8)
    stepid[..., :4] = be(tasks.astype(np.uint32)).view(np.uint8).reshape(-1, 1, 4)
    stepid[..., 4:12] = be(counters.astype(np.uint64)).view(np.uint8).reshape(-1, 1, 8)
    stepid[..., 12:] = be(self._steps.astype(np.uint64)).view(np.uint8).reshape(1, T, 8)

    return {
        "vector": np.take(store.obs, idx, axis=0),
        "action": np.take(store.action, idx, axis=0),
        "reward": reward,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
        "stepid": stepid,
        "consec": np.zeros((batch_size, T), np.int32),
        "task_id": np.repeat(tasks[:, None].astype(np.int32), T, 1),
        "task_axes": np.repeat(self.axes[tasks][:, None], T, 1),
    }

  def stats(self):
    return {
        "tasks": len(self.tasks),
        "schedule_mixed": float(self.schedule == "mixed"),
        "schedule_sequential": float(self.schedule == "sequential"),
        "current_task_index": float(self._task_order[self._task_position])
            if self._task_order else -1.0,
        "current_task_batches": float(self._task_batch_count),
        "transitions_m": float(self.lengths.sum() / 1e6),
    }

  def _choose_tasks(self, count):
    if self.sampling == "uniform_task":
      return self.rng.integers(0, len(self.tasks), count)
    return self.rng.choice(len(self.tasks), count, p=self.transition_probs)

  def _reset_task_order(self):
    self._task_order = list(range(len(self.tasks)))
    if self.shuffle_tasks:
      self.rng.shuffle(self._task_order)

  def _choose_sequential_task(self):
    task_index = self._task_order[self._task_position]
    self._task_batch_count += 1
    if self._task_batch_count >= self.batches_per_task:
      self._task_batch_count = 0
      self._task_position += 1
      if self._task_position >= len(self._task_order):
        self._task_position = 0
        self._reset_task_order()
    return task_index


def _get(mapping, key, default=None):
  if mapping is None:
    return default
  if hasattr(mapping, "get"):
    return mapping.get(key, default)
  return getattr(mapping, key, default)


def _as_bool(value):
  if isinstance(value, str):
    return value.lower() in ("1", "true", "yes", "y", "on")
  return bool(value)


def make_datasets(
    data_config, train_length, report_length, seed=0, logdir=None):
  root = _get(data_config, "root")
  train = _get(data_config, "train")
  test = _get(data_config, "test")
  allow_overlap = _as_bool(_get(data_config, "allow_overlap", False))
  _, train_tasks, test_tasks = selection.resolve_selection(
      root, train, test, allow_overlap)
  selection.write_resolved(
      logdir if logdir is not None else pathlib.Path("."),
      train_tasks, test_tasks)
  sampling_config = _get(data_config, "sampling", {})
  sampling = _get(sampling_config, "mode", "uniform_task")
  schedule = _get(sampling_config, "schedule", "mixed")
  eval_schedule = _get(sampling_config, "eval_schedule", "mixed")
  batches_per_task = _get(sampling_config, "batches_per_task", 1000)
  shuffle_tasks = _get(sampling_config, "shuffle_tasks", False)
  store = Store(train_tasks + test_tasks)
  train_dataset = OfflineCompDataset(
      train_tasks, train_length, seed=seed, sampling=sampling,
      schedule=schedule, batches_per_task=batches_per_task,
      shuffle_tasks=shuffle_tasks, store=store)
  train_report_dataset = OfflineCompDataset(
      train_tasks, report_length, seed=seed + 2, sampling=sampling,
      schedule=schedule, batches_per_task=batches_per_task,
      shuffle_tasks=shuffle_tasks, store=store)
  test_report_dataset = OfflineCompDataset(
      test_tasks, report_length, seed=seed + 1, sampling=sampling,
      schedule=eval_schedule, batches_per_task=batches_per_task,
      shuffle_tasks=shuffle_tasks, store=store)
  return (
      train_dataset, train_report_dataset, test_report_dataset,
      train_tasks, test_tasks)
