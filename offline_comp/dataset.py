import itertools
import pathlib
import threading

import elements
import numpy as np

from . import selection
from . import tasks as tasks_mod


_H5_LOCK = threading.Lock()


class OfflineCompDataset:

  def __init__(
      self, tasks, sequence_length, seed=0, sampling="uniform_task",
      schedule="mixed", batches_per_task=1000, shuffle_tasks=False):
    self.tasks = list(tasks)
    self.sequence_length = int(sequence_length)
    self.sampling = sampling
    self.schedule = schedule
    self.batches_per_task = int(batches_per_task)
    self.shuffle_tasks = _as_bool(shuffle_tasks)
    self.rng = np.random.default_rng(seed)
    self.files = []
    self.lengths = []
    self.obs_shape = None
    self.action_shape = None
    self._counter = itertools.count()
    self._task_position = 0
    self._task_batch_count = 0
    self._task_order = []
    self._closed = False

    import h5py
    with _H5_LOCK:
      for task in self.tasks:
        handle = h5py.File(task.path, "r")
        try:
          self._validate_file(task, handle)
        except Exception:
          handle.close()
          raise
        self.files.append(handle)
        self.lengths.append(int(handle["observations"].shape[0]))

    self.lengths = np.asarray(self.lengths, np.int64)
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
    weights = self.lengths.astype(np.float64)
    self.transition_probs = weights / weights.sum()
    self._reset_task_order()

  @classmethod
  def from_config(cls, data_config, split, sequence_length, seed=0):
    root = _get(data_config, "root")
    if not root:
      raise ValueError("data.root must point at an extracted Dryad archive")
    train = _get(data_config, "train")
    test = _get(data_config, "test")
    allow_overlap = _as_bool(_get(data_config, "allow_overlap", False))
    _, train_tasks, test_tasks = selection.resolve_selection(
        root, train, test, allow_overlap)
    selected = train_tasks if split == "train" else test_tasks
    sampling_config = _get(data_config, "sampling", {})
    sampling = _get(sampling_config, "mode", "uniform_task")
    schedule = _get(sampling_config, "schedule", "mixed")
    if split != "train":
      schedule = _get(sampling_config, "eval_schedule", "mixed")
    batches_per_task = _get(sampling_config, "batches_per_task", 1000)
    shuffle_tasks = _get(sampling_config, "shuffle_tasks", False)
    return cls(
        selected, sequence_length, seed=seed, sampling=sampling,
        schedule=schedule, batches_per_task=batches_per_task,
        shuffle_tasks=shuffle_tasks)

  def close(self):
    with _H5_LOCK:
      if self._closed:
        return
      self._closed = True
      for handle in self.files:
        try:
          handle.close()
        except Exception:
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
      rows = [self._sample_one() for _ in range(batch_size)]
    elif self.schedule == "sequential":
      task_index = self._choose_sequential_task()
      rows = [self._sample_one(task_index) for _ in range(batch_size)]
    else:
      raise NotImplementedError(self.schedule)
    return {
        key: np.stack([row[key] for row in rows], axis=0)
        for key in rows[0].keys()
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

  def _validate_file(self, task, handle):
    required = ("observations", "actions", "rewards", "terminals", "timeouts")
    missing = [key for key in required if key not in handle]
    if missing:
      raise ValueError(f"{task.path} is missing HDF5 keys: {missing}")
    obs_shape = tuple(handle["observations"].shape[1:])
    action_shape = tuple(handle["actions"].shape[1:])
    if self.obs_shape is None:
      self.obs_shape = obs_shape
      self.action_shape = action_shape
    if obs_shape != self.obs_shape or action_shape != self.action_shape:
      raise ValueError(
          f"Incompatible shapes in {task.name}: obs {obs_shape}, "
          f"action {action_shape}; expected {self.obs_shape}, "
          f"{self.action_shape}")

  def _choose_task(self):
    if self.sampling == "uniform_task":
      return int(self.rng.integers(0, len(self.files)))
    if self.sampling == "uniform_transition":
      return int(self.rng.choice(len(self.files), p=self.transition_probs))
    raise ValueError(
        "sampling.mode must be 'uniform_task' or 'uniform_transition', "
        f"got {self.sampling!r}")

  def _reset_task_order(self):
    self._task_order = list(range(len(self.files)))
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

  def _sample_one(self, task_index=None):
    if task_index is None:
      task_index = self._choose_task()
    task = self.tasks[task_index]
    axes = np.array([
        tasks_mod.ROBOTS.index(task.robot),
        tasks_mod.OBJECTS.index(task.obj),
        tasks_mod.OBSTACLES.index(task.obstacle),
        tasks_mod.OBJECTIVES.index(task.objective),
    ], np.int32)
    handle = self.files[task_index]
    length = self.lengths[task_index]
    start = int(self.rng.integers(0, length - self.sequence_length + 1))
    stop = start + self.sequence_length

    with _H5_LOCK:
      if self._closed:
        raise StopIteration
      obs = np.asarray(handle["observations"][start:stop], np.float32)
      action = np.asarray(handle["actions"][start:stop], np.float32)
      raw_reward = np.asarray(
          handle["rewards"][start:stop], np.float32).reshape(-1)
      raw_terminal = np.asarray(
          handle["terminals"][start:stop], bool).reshape(-1)
      raw_timeout = np.asarray(
          handle["timeouts"][start:stop], bool).reshape(-1)

    reward = np.zeros((self.sequence_length,), np.float32)
    is_last = np.zeros((self.sequence_length,), bool)
    is_terminal = np.zeros((self.sequence_length,), bool)
    if self.sequence_length > 1:
      reward[1:] = raw_reward[:-1]
      is_terminal[1:] = raw_terminal[:-1]
      is_last[1:] = raw_terminal[:-1] | raw_timeout[:-1]
    is_first = np.zeros((self.sequence_length,), bool)
    is_first[0] = True
    is_first[1:] = is_last[:-1]

    stepid = np.zeros((self.sequence_length, 20), np.uint8)
    counter = next(self._counter)
    prefix = task_index.to_bytes(4, "big") + counter.to_bytes(8, "big")
    for index in range(self.sequence_length):
      stepid[index] = np.frombuffer(
          prefix + index.to_bytes(8, "big"), np.uint8)

    return {
        "vector": obs,
        "action": action,
        "reward": reward,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
        "stepid": stepid,
        "consec": np.zeros((self.sequence_length,), np.int32),
        "task_id": np.full((self.sequence_length,), task_index, np.int32),
        "task_axes": np.tile(axes, (self.sequence_length, 1)),
    }


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
  train_dataset = OfflineCompDataset(
      train_tasks, train_length, seed=seed, sampling=sampling,
      schedule=schedule, batches_per_task=batches_per_task,
      shuffle_tasks=shuffle_tasks)
  train_report_dataset = OfflineCompDataset(
      train_tasks, report_length, seed=seed + 2, sampling=sampling,
      schedule=schedule, batches_per_task=batches_per_task,
      shuffle_tasks=shuffle_tasks)
  test_report_dataset = OfflineCompDataset(
      test_tasks, report_length, seed=seed + 1, sampling=sampling,
      schedule=eval_schedule, batches_per_task=batches_per_task,
      shuffle_tasks=shuffle_tasks)
  return (
      train_dataset, train_report_dataset, test_report_dataset,
      train_tasks, test_tasks)