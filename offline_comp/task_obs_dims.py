"""Find observation dims that reveal the task, for --agent.mask_obs.

CompoSuite envs built with use_task_id_obs=True append a one-hot indicator of
(robot, object, obstacle, objective) to the observation vector, and the offline
datasets carry the same vector. A model that reads it gets the task label even
when `task_id` is kept away from the encoder. This script reports dims that
are constant within every task, binary, and differ across tasks.

  python -m offline_comp.task_obs_dims --root data/expert-panda-offline-comp-data
  python -m offline_comp.task_obs_dims --root ... --rows 20000 --show

Inspect the reported dims before masking them: a binary per-task flag that
describes physical state (rather than the task label) would also be listed.
"""

import argparse

import numpy as np

from .dataset import detect_task_obs_dims, dims_to_spec
from .tasks import discover


class _Sample:
  """Store-like view (obs, span) over the first `rows` rows of each task."""

  def __init__(self, tasks, rows):
    import h5py
    chunks, self.span, start = [], {}, 0
    for task in tasks:
      with h5py.File(task.path, "r") as f:
        n = int(f["observations"].shape[0])
        n = min(n, rows) if rows > 0 else n
        chunks.append(np.asarray(f["observations"][:n], np.float32))
      self.span[task.path] = (start, n)
      start += n
    self.obs = np.concatenate(chunks, 0)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--root", required=True)
  parser.add_argument("--rows", type=int, default=20000,
                      help="rows read per task (0 = all)")
  parser.add_argument("--show", action="store_true",
                      help="print the value of each reported dim per task")
  args = parser.parse_args(argv)

  tasks = discover(args.root)
  if not tasks:
    raise SystemExit(f"No task folders with data.hdf5 under {args.root}")
  store = _Sample(tasks, args.rows)
  varying, constant = detect_task_obs_dims(store, tasks)
  dim = store.obs.shape[-1]
  print(f"{len(tasks)} tasks, observation dim {dim}")
  print(f"task-identifying dims (vary across tasks): "
        f"{dims_to_spec(varying) or 'none'}")
  print(f"binary dims constant for all tasks:         "
        f"{dims_to_spec(constant) or 'none'}")
  if varying:
    print(f"\n--agent.mask_obs {dims_to_spec(varying + constant)}")
  if args.show and (varying or constant):
    dims = sorted(varying + constant)
    print("\ntask" + "".join(f" {d:>4}" for d in dims))
    for task in tasks:
      start, n = store.span[task.path]
      row = store.obs[start, dims]
      print(task.name + "".join(f" {int(v):>4}" for v in row))


if __name__ == "__main__":
  main()
