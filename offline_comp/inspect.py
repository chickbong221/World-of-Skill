import argparse
import collections

import ruamel.yaml as yaml

from . import selection
from .tasks import discover


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True)
  parser.add_argument("--config")
  args = parser.parse_args(argv)

  tasks = discover(args.root)
  print(f"Tasks: {len(tasks)}")
  grouped = collections.defaultdict(set)
  for task in tasks:
    grouped["robots"].add(task.robot)
    grouped["objects"].add(task.obj)
    grouped["obstacles"].add(task.obstacle)
    grouped["objectives"].add(task.objective)
  for key in ("robots", "objects", "obstacles", "objectives"):
    print(f"{key}: {', '.join(sorted(grouped[key]))}")

  if args.config:
    config = yaml.YAML(typ="safe").load(open(args.config, "r"))
    data = config.get("data", config)
    _, train, test = selection.resolve_selection(
        args.root, data.get("train"), data.get("test"),
        data.get("allow_overlap", False))
    print(f"Resolved train tasks: {len(train)}")
    for task in train:
      print(f"  train {task.name}")
    print(f"Resolved test tasks: {len(test)}")
    for task in test:
      print(f"  test  {task.name}")


if __name__ == "__main__":
  main()
