import json
import pathlib

from . import tasks as tasklib


def _plain(value):
  if hasattr(value, "items"):
    return {k: _plain(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain(x) for x in value]
  return value


def _get(mapping, key, default=None):
  if mapping is None:
    return default
  if hasattr(mapping, "get"):
    return mapping.get(key, default)
  return getattr(mapping, key, default)


def _as_list(value):
  if value is None:
    return None
  if isinstance(value, str):
    return [value]
  return list(value)


def _filter_tasks(available, spec):
  spec = _plain(spec or {})
  exact = _get(spec, "tasks")
  include = _get(spec, "include")
  exclude = _get(spec, "exclude")

  if exact:
    requested = [tasklib.task_from_sequence(x) for x in exact]
    by_tuple = {x.tuple: x for x in available}
    missing = [x.name for x in requested if x.tuple not in by_tuple]
    if missing:
      raise ValueError(f"Requested tasks are not available: {missing}")
    selected = [by_tuple[x.tuple] for x in requested]
  else:
    selected = list(available)
    if include:
      include = _plain(include)
      axes = {
          "robots": ("robot", tasklib.ROBOTS),
          "objects": ("obj", tasklib.OBJECTS),
          "obstacles": ("obstacle", tasklib.OBSTACLES),
          "objectives": ("objective", tasklib.OBJECTIVES),
      }
      for key, (attr, _) in axes.items():
        values = _as_list(_get(include, key))
        if values is None:
          continue
        values = {tasklib.normalize(x) for x in values}
        selected = [x for x in selected if getattr(x, attr) in values]

  if exclude:
    exclude = _plain(exclude)
    blocked = {tasklib.task_from_sequence(x).tuple for x in _get(exclude, "tasks", [])}
    selected = [x for x in selected if x.tuple not in blocked]

  return sorted(selected)


def resolve_selection(root, train, test, allow_overlap=False):
  available = tasklib.discover(root)
  if not available:
    raise ValueError(f"No task folders with data.hdf5 found under {root!r}")

  train_tasks = _filter_tasks(available, train)
  test_tasks = _filter_tasks(available, test)
  if not train_tasks:
    raise ValueError("Train selection resolved to zero tasks")
  if not test_tasks:
    raise ValueError("Test selection resolved to zero tasks")

  overlap = sorted(set(x.tuple for x in train_tasks) & set(x.tuple for x in test_tasks))
  if overlap and not allow_overlap:
    names = ["_".join(x) for x in overlap]
    raise ValueError(f"Train/test selections overlap: {names}")
  return available, train_tasks, test_tasks


def write_resolved(logdir, train_tasks, test_tasks):
  logdir = pathlib.Path(str(logdir))
  logdir.mkdir(parents=True, exist_ok=True)
  payloads = {
      "resolved_train_tasks.json": train_tasks,
      "resolved_test_tasks.json": test_tasks,
  }
  for filename, selected in payloads.items():
    data = [
        {
            "name": task.name,
            "robot": task.robot,
            "object": task.obj,
            "obstacle": task.obstacle,
            "objective": task.objective,
            "path": task.path,
        }
        for task in selected
    ]
    (logdir / filename).write_text(json.dumps(data, indent=2) + "\n")
