"""
Plot task-by-expert routing-mass heatmaps for MoSS, one figure per split
(train / report / eval), with a discrete (binned) color scale. Adapted from
MoDULE's plot_mass_diff_heatmap.py.

The values are the mean soft router mass pi each task sends to each expert,
logged as <split>/moss_route/<task>/e<m> (embodied/run/train_offline.py,
route_mass). Each row sums to 1.

Input, either:
  --jsonl  logdir/<run>/metrics.jsonl  (local run log), or
  --csv    a wandb "download CSV" export with columns
           _step, <split>/moss_route/<task>/e<m>, ...
For each split we take the last logged values at or before --step (default:
the last step that split was logged).

Usage:
    python scripts/plot_route_mass_heatmap.py \
        --jsonl logdir/moss_lsep_nobal_msp001_e8k2/metrics.jsonl \
        --out_dir figures/route_mass --scale log --cmap Reds --annotate
"""

import argparse
import csv
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

COL_RE = re.compile(
    r"^(?P<split>[a-z_]+)/moss_route/(?P<task>[A-Za-z0-9_]+)/e(?P<expert>\d+)$")

SPLIT_LABELS = {
    "train": "Train tasks (training batches)",
    "report": "Train tasks (eval mode)",
    "eval": "Held-out test tasks",
}

# Row order: object, then obstacle (CompoSuite task names are
# <robot>_<object>_<obstacle>_<objective>).
OBJECT_ORDER = ["Box", "Dumbbell", "Hollowbox", "Plate"]
OBSTACLE_ORDER = ["None", "GoalWall", "ObjectDoor", "ObjectWall"]

# 1-2-5 sequence for clean log-spaced bin edges (1e-4, 2e-4, 5e-4, 1e-3, ...).
_LOG_STEPS = [1, 2, 5]


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--jsonl", help="Path to a run's metrics.jsonl")
  src.add_argument("--csv", help="Path to a wandb CSV export")
  p.add_argument("--step", type=float, default=None,
                 help="Step to visualize (default: last step logged per split)")
  p.add_argument("--splits", default="train,report,eval",
                 help="Comma-separated splits to plot (skipped if not logged)")
  p.add_argument("--out_dir", default="figures/route_mass",
                 help="Directory to write output figures")
  p.add_argument("--cmap", default="viridis",
                 help="Base sequential colormap to discretize (e.g. 'viridis', 'Reds')")
  p.add_argument("--scale", choices=["log", "linear"], default="log",
                 help="Bin edges spaced in log or linear space. Log shows the near-zero "
                      "mass of unused experts; linear shows the dominant expert.")
  p.add_argument("--vmin", type=float, default=1e-4,
                 help="Log scale only: values below this go into the lowest bin")
  p.add_argument("--n_bins", type=int, default=None,
                 help="Linear scale only: target number of bins (default 10)")
  p.add_argument("--fmt", default="pdf", choices=["pdf", "png"], help="Output file format")
  p.add_argument("--dpi", type=int, default=300)
  p.add_argument("--annotate", action="store_true", default=False,
                 help="Write the numeric value in each cell")
  return p.parse_args()


def _num(x):
  try:
    v = float(x)
  except (TypeError, ValueError):
    return None
  return None if math.isnan(v) else v


def load_rows(args):
  """Return a list of (step, {key: value}) restricted to moss_route keys."""
  rows = []
  if args.jsonl:
    with open(args.jsonl) as f:
      for line in f:
        r = json.loads(line)
        vals = {k: v for k, v in r.items() if COL_RE.match(k)}
        if vals:
          rows.append((float(r["step"]), vals))
  else:
    with open(args.csv, newline="") as f:
      for r in csv.DictReader(f):
        step = _num(r.get("_step", r.get("step")))
        vals = {k: _num(v) for k, v in r.items() if COL_RE.match(k or "")}
        vals = {k: v for k, v in vals.items() if v is not None}
        if step is not None and vals:
          rows.append((step, vals))
  rows.sort(key=lambda x: x[0])
  return rows


def last_values(rows, split, target_step):
  """Last non-missing value per key of this split at or before target_step."""
  prefix = f"{split}/moss_route/"
  logged = [s for s, vals in rows if any(k.startswith(prefix) for k in vals)]
  if not logged:
    return {}, None
  if target_step is None:
    target_step = logged[-1]
  last, used = {}, None
  for step, vals in rows:
    if step > target_step:
      break
    for k, v in vals.items():
      if k.startswith(prefix):
        last[k] = v
        used = step
  return last, used


def task_sort_key(task):
  parts = task.split("_")
  obj = parts[1] if len(parts) > 1 else task
  obs = parts[2] if len(parts) > 2 else ""
  oi = OBJECT_ORDER.index(obj) if obj in OBJECT_ORDER else len(OBJECT_ORDER)
  bi = OBSTACLE_ORDER.index(obs) if obs in OBSTACLE_ORDER else len(OBSTACLE_ORDER)
  return (oi, obj, bi, obs, task)


def task_label(task):
  """Panda_Box_GoalWall_PickPlace -> Box / GoalWall."""
  parts = task.split("_")
  return " / ".join(parts[1:3]) if len(parts) >= 3 else task


def build_matrix(values):
  """Return (tasks, experts, matrix [task x expert])."""
  cells = {}
  for k, v in values.items():
    m = COL_RE.match(k)
    cells[(m.group("task"), int(m.group("expert")))] = v
  tasks = sorted({t for t, _ in cells}, key=task_sort_key)
  experts = sorted({e for _, e in cells})
  mat = np.full((len(tasks), len(experts)), np.nan)
  for i, t in enumerate(tasks):
    for j, e in enumerate(experts):
      mat[i, j] = cells.get((t, e), np.nan)
  return tasks, experts, mat


def nice_log_boundaries(vmin, vmax):
  """Clean 1-2-5-spaced bin edges in log space covering [vmin, vmax]."""
  lo_exp = int(np.floor(np.log10(vmin)))
  hi_exp = int(np.ceil(np.log10(vmax)))
  edges = sorted({s * 10.0 ** e for e in range(lo_exp, hi_exp + 1) for s in _LOG_STEPS})
  lo = max([i for i, e in enumerate(edges) if e <= vmin], default=0)
  hi = min([i for i, e in enumerate(edges) if e >= vmax], default=len(edges) - 1)
  return np.array(edges[lo:hi + 1])


def nice_linear_boundaries(vmin, vmax, n_bins):
  """Clean round-number bin edges in linear space (e.g. steps of 0.1)."""
  raw_step = (vmax - vmin) / n_bins
  magnitude = 10 ** np.floor(np.log10(raw_step))
  for mult in [1, 2, 2.5, 5, 10]:
    step = mult * magnitude
    if step >= raw_step:
      break
  lo = np.floor(vmin / step) * step
  hi = np.ceil(vmax / step) * step
  n = int(round((hi - lo) / step)) + 1
  return np.linspace(lo, hi, n)


def _fmt_tick(b):
  """Compact tick label: '1e-4' instead of '1.0e-04', '0.05' stays plain."""
  if b == 0 or (0.001 <= abs(b) < 1000):
    return f"{b:g}"
  mantissa, exp = f"{b:.0e}".split("e")
  return f"{mantissa}e{int(exp)}"


def plot_split(tasks, experts, mat, split, step, boundaries, args, out_path):
  plt.rcParams.update({
      "font.size": 11,
      "axes.edgecolor": "#333333",
      "axes.linewidth": 0.8,
  })
  vals = mat.copy()
  if args.scale == "log":
    # Clip into the bin range so near-zero mass lands in the lowest bin.
    vals = np.clip(vals, boundaries[0], boundaries[-1])

  n_colors = len(boundaries) - 1
  base_cmap = plt.get_cmap(args.cmap, n_colors)
  cmap = ListedColormap(base_cmap(np.linspace(0, 1, n_colors)))
  norm = BoundaryNorm(boundaries, ncolors=n_colors)

  height = max(2.2, 0.32 * len(tasks) + 1.0)
  fig, ax = plt.subplots(figsize=(5.8, height))
  im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")

  # Thin white gridlines between cells.
  ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True)
  ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
  ax.grid(which="minor", color="white", linewidth=1.5)
  ax.tick_params(which="minor", bottom=False, left=False)

  ax.set_xticks(range(len(experts)))
  ax.set_xticklabels([str(e) for e in experts])
  ax.set_yticks(range(len(tasks)))
  ax.set_yticklabels([task_label(t) for t in tasks])
  ax.set_xlabel("Expert index")
  ax.set_title(f"{SPLIT_LABELS.get(split, split)}, step {int(step):,}", fontsize=11)

  if args.annotate:
    mid = boundaries[len(boundaries) // 2]
    for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        v = mat[i, j]
        if not np.isnan(v):
          color = "white" if v > mid else "black"
          text = f"{v:.2f}" if v >= 0.01 or args.scale == "linear" else _fmt_tick(
              float(f"{v:.0e}"))
          ax.text(j, i, text, ha="center", va="center",
                  fontsize=6.5, color=color)

  cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, aspect=22,
                      boundaries=boundaries, ticks=boundaries)
  cbar.ax.tick_params(labelsize=8, length=2, pad=3)
  cbar.ax.set_yticklabels([_fmt_tick(b) for b in boundaries])
  cbar.outline.set_linewidth(0.6)
  cbar.set_label("Routing mass", fontsize=9)

  fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved: {out_path}")


def main():
  args = parse_args()
  rows = load_rows(args)
  if not rows:
    raise SystemExit("No <split>/moss_route/<task>/e<m> columns found.")

  splits = {}
  for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
    values, step = last_values(rows, split, args.step)
    if not values:
      print(f"Skipping {split}: not logged at or before step {args.step}")
      continue
    splits[split] = (step, *build_matrix(values))

  # Shared bin edges across splits so the figures are directly comparable.
  all_vals = np.concatenate([m.ravel() for _, _, _, m in splits.values()])
  all_vals = all_vals[~np.isnan(all_vals)]
  vmax = float(all_vals.max())
  if args.scale == "log":
    boundaries = nice_log_boundaries(args.vmin, max(vmax, args.vmin * 10))
  else:
    boundaries = nice_linear_boundaries(0.0, vmax, args.n_bins or 10)

  os.makedirs(args.out_dir, exist_ok=True)
  for split, (step, tasks, experts, mat) in splits.items():
    out_path = os.path.join(
        args.out_dir, f"route_mass_{split}_step{int(step)}_{args.scale}.{args.fmt}")
    plot_split(tasks, experts, mat, split, step, boundaries, args, out_path)


if __name__ == "__main__":
  main()
