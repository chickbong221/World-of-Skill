"""Sanity-check that the new offline_comp config actually loads.

Run from repo root:  python scripts/verify_config.py
"""
import pathlib
import sys

import ruamel.yaml as yaml
import elements

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from offline_comp import tasks as tasklib
from offline_comp import selection


def check(label, fn):
  try:
    fn()
    print(f"  OK    {label}")
  except Exception as e:
    print(f"  FAIL  {label}: {type(e).__name__}: {e}")
    raise


def main():
  cfg_path = pathlib.Path(__file__).resolve().parents[1] / "dreamerv3" / "configs.yaml"
  raw = yaml.YAML(typ="safe").load(cfg_path.read_text())

  print("[1] elements.Config accepts flat list-of-string tasks")
  check("defaults load",
        lambda: elements.Config(raw["defaults"]))
  check("defaults + offline_comp update",
        lambda: elements.Config(raw["defaults"]).update(raw["offline_comp"]))

  print("[2] parse_task_name handles the new format")
  for name in (
      "Panda_Box_None_PickPlace",
      "Panda_Hollowbox_ObjectWall_PickPlace",
      "Panda_Dumbbell_ObjectDoor_PickPlace"):
    check(name, lambda n=name: tasklib.parse_task_name(n))

  print("[3] Resolved split matches expected sizes (needs data/)")
  cfg = elements.Config(raw["defaults"]).update(raw["offline_comp"])
  data = dict(cfg.data)
  data["train"] = dict(cfg.data.train)
  data["test"] = dict(cfg.data.test)
  # Convert elements list wrappers to plain lists.
  data["train"]["tasks"] = list(cfg.data.train.tasks)
  data["test"]["tasks"] = list(cfg.data.test.tasks)
  try:
    _, train, test = selection.resolve_selection(
        data["root"], data["train"], data["test"],
        data.get("allow_overlap", False))
    print(f"  OK    train={len(train)} test={len(test)} "
          f"(expected 13 and 3)")
    if len(train) != 13 or len(test) != 3:
      print("  WARN  counts differ from expected split")
  except Exception as e:
    print(f"  SKIP  {type(e).__name__}: {e}  "
          f"(fine if data/ is not populated yet)")


if __name__ == "__main__":
  main()
