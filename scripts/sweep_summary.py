"""Summarize a MoSS sweep (scripts/sweep_router.sh or sweep_msp_msep.sh).

  python scripts/sweep_summary.py
  python scripts/sweep_summary.py --root logdir/sweep_msp_msep \
      --include logdir/moss_lsep_nobal_msp001_e8k2 logdir/moss_expert_debug_e8k2

Ranks trials by the VALIDATION world-model loss val/loss/dyn (held-out rows
of the train tasks, data.val_frac; mean of the last 3 reports, lower is
better) and prints the grid of it. eval/ (the held-out recombination tasks)
is shown but never used for ranking: selecting on it leaks the test split.
Runs without a val/ split fall back to report/ (train split) with a warning.
Standard library only.
"""
import argparse
import json
import math
import pathlib
import re

NAME_RE = re.compile(r'msp(?P<msp>[^_]+)_msep(?P<msep>.+)$')
NAME_RE_ROUTER = re.compile(r'mcomp(?P<mcomp>[^_]+)_msep(?P<msep>.+)$')
WM_END = 25000  # last train log of the world-model phase


def load(path):
  rows = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def series(rows, key):
  return [r[key] for r in rows if key in r]


def last_mean(rows, key, n=3):
  vals = series(rows, key)[-n:]
  return sum(vals) / len(vals) if vals else math.nan


def at_step(rows, key, step):
  cand = [r for r in rows if key in r]
  if not cand:
    return math.nan
  return min(cand, key=lambda r: abs(r['step'] - step))[key]


def maximum(rows, key):
  vals = series(rows, key)
  return max(vals) if vals else math.nan


def summarize(logdir):
  rows = load(logdir / 'metrics.jsonl')
  cfg = (logdir / 'config.yaml').read_text() if (
      logdir / 'config.yaml').exists() else ''
  grab = lambda k: (re.search(rf'\b{k}: ([0-9.e-]+)', cfg) or [None, '?'])[1]
  msp, msep, mcomp = grab('msp'), grab('msep'), grab('mcomp')
  m = NAME_RE.search(logdir.name) or NAME_RE_ROUTER.search(logdir.name)
  if m:
    groups = m.groupdict()
    msp = groups.get('msp', msp)
    mcomp = groups.get('mcomp', mcomp)
    msep = groups['msep']
  msep = '0' if msep == '?' and cfg else msep  # runs from before L_sep
  mcomp = '0' if mcomp == '?' and cfg else mcomp  # runs from before L_comp
  bc = (re.search(r'bc_scale: ([0-9.e-]+)', cfg) or [None, '?'])[1]
  val_dyn = last_mean(rows, 'val/loss/dyn')
  has_val = not math.isnan(val_dyn)
  return {
      'run': logdir.name,
      'msp': msp,
      'mcomp': mcomp,
      'msep': msep,
      'bc': bc,
      'done': (logdir / 'done').exists(),
      'step': rows[-1]['step'] if rows else 0,
      'select': val_dyn if has_val else last_mean(rows, 'report/loss/dyn'),
      'select_on': 'val' if has_val else 'train',
      'val_dyn': val_dyn,
      'eval_dyn': last_mean(rows, 'eval/loss/dyn'),
      'eval_vec': last_mean(rows, 'eval/loss/vector'),
      'report_dyn': last_mean(rows, 'report/loss/dyn'),
      'mi_train': at_step(rows, 'train/moss_sep_mi_norm', WM_END),
      'mi_eval': last_mean(rows, 'eval/moss_sep_mi_norm'),
      'dead': at_step(rows, 'train/moss_dead_experts', WM_END),
      'eff': at_step(rows, 'train/moss_eff_experts', WM_END),
      'tr_ret': series(rows, 'env_eval/train/return')[-1:] or [math.nan],
      'te_ret': series(rows, 'env_eval/test/return')[-1:] or [math.nan],
      'te_ret_max': maximum(rows, 'env_eval/test/return'),
      'tr_succ': maximum(rows, 'env_eval/train/success_once'),
      'te_succ': maximum(rows, 'env_eval/test/success_once'),
  }


def fmt(x, digits=3):
  if isinstance(x, list):
    x = x[0]
  if isinstance(x, float):
    return '—' if math.isnan(x) else f'{x:.{digits}f}'
  return str(x)


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--root', default='logdir/sweep_msp_msep')
  parser.add_argument('--include', nargs='*', default=[],
                      help='extra run logdirs shown as references')
  args = parser.parse_args()

  root = pathlib.Path(args.root)
  dirs = sorted(d for d in root.glob('*') if (d / 'metrics.jsonl').exists())
  refs = [pathlib.Path(d) for d in args.include
          if (pathlib.Path(d) / 'metrics.jsonl').exists()]
  if not dirs and not refs:
    print(f'No runs with metrics.jsonl under {root}')
    return

  trials = [summarize(d) for d in dirs]
  references = [dict(summarize(d), run=f'(ref) {d.name}') for d in refs]
  if any(r['select_on'] != 'val' for r in trials):
    print('WARNING: some runs have no val/ split (data.val_frac = 0); they '
          'are ranked on the train-split report instead. eval/ is never '
          'used for ranking.' + chr(10))
  key = lambda r: (math.isnan(r['select']), r['select'])
  table = sorted(trials, key=key) + references

  cols = [
      ('run', 'run'), ('msp', 'msp'), ('mcomp', 'mcomp'), ('msep', 'msep'),
      ('bc', 'bc'), ('step', 'step'), ('select', 'select dyn ↓'),
      ('select_on', 'on'), ('eval_dyn', 'eval dyn (test)'),
      ('eval_vec', 'eval vector'), ('report_dyn', 'train dyn'),
      ('mi_train', 'MI train'), ('mi_eval', 'MI eval'),
      ('dead', 'dead'), ('eff', 'eff'),
      ('tr_ret', 'train ret'), ('te_ret', 'test ret'),
      ('te_ret_max', 'test ret max'),
      ('tr_succ', 'train succ max'), ('te_succ', 'test succ max'),
  ]
  print('| ' + ' | '.join(c[1] for c in cols) + ' |')
  print('|' + '---|' * len(cols))
  for r in table:
    cells = []
    for k, _ in cols:
      v = r[k]
      if k == 'step' and not r['done'] and not r['run'].startswith('(ref)'):
        v = f'{v} (running)'
      cells.append(fmt(v, 2 if k in ('dead', 'eff') else 3))
    print('| ' + ' | '.join(cells) + ' |')

  if trials:
    num = lambda x: float(x) if x not in ('?', '') else math.nan
    rowkey = 'mcomp' if len({r['mcomp'] for r in trials}) > 1 else 'msp'
    known = [r for r in trials if '?' not in (r[rowkey], r['msep'])]
    rowvals = sorted({r[rowkey] for r in known}, key=num)
    mseps = sorted({r['msep'] for r in known}, key=num)
    grid = {(r[rowkey], r['msep']): r['select'] for r in known}
    print(chr(10) + 'selection loss (val/loss/dyn, mean of last 3 reports), '
          f'rows {rowkey}, columns msep')
    print(f'| {rowkey} / msep | ' + ' | '.join(mseps) + ' |')
    print('|---|' + '---:|' * len(mseps))
    for p in rowvals:
      print(f'| {p} | ' + ' | '.join(
          fmt(grid.get((p, s), math.nan)) for s in mseps) + ' |')

if __name__ == '__main__':
  main()
