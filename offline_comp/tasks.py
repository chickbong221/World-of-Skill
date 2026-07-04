import dataclasses
import itertools
import pathlib


ROBOTS = ("IIWA", "Jaco", "Kinova3", "Panda")
OBJECTS = ("Box", "Dumbbell", "Plate", "Hollowbox")
OBSTACLES = ("None", "GoalWall", "ObjectDoor", "ObjectWall")
OBJECTIVES = ("PickPlace", "Push", "Shelf", "Trashcan")


_ALIASES = {
    "hollow_box": "Hollowbox",
    "hollowbox": "Hollowbox",
    "hollow box": "Hollowbox",
    "pick_place": "PickPlace",
    "pickplace": "PickPlace",
    "pick and place": "PickPlace",
    "objectdoor": "ObjectDoor",
    "object_door": "ObjectDoor",
    "door": "ObjectDoor",
    "goalwall": "GoalWall",
    "goal_wall": "GoalWall",
    "objectwall": "ObjectWall",
    "object_wall": "ObjectWall",
    "none": "None",
    "null": "None",
}


@dataclasses.dataclass(frozen=True, order=True)
class Task:
  robot: str
  obj: str
  obstacle: str
  objective: str
  path: str = ""

  @property
  def name(self):
    return f"{self.robot}_{self.obj}_{self.obstacle}_{self.objective}"

  @property
  def tuple(self):
    return (self.robot, self.obj, self.obstacle, self.objective)

  def with_path(self, path):
    return dataclasses.replace(self, path=str(path))


def normalize(value):
  if value is None:
    return "None"
  text = str(value).strip()
  key = text.lower().replace("-", "_")
  if key in _ALIASES:
    return _ALIASES[key]
  for choices in (ROBOTS, OBJECTS, OBSTACLES, OBJECTIVES):
    for choice in choices:
      if key == choice.lower():
        return choice
  return text


def parse_task_name(name):
  parts = pathlib.Path(name).name.split("_")
  if len(parts) != 4:
    raise ValueError(
        f"Expected task folder '<robot>_<object>_<obstacle>_<objective>', "
        f"got {name!r}")
  task = Task(*(normalize(x) for x in parts))
  validate_task(task)
  return task


def validate_task(task):
  values = (task.robot, task.obj, task.obstacle, task.objective)
  choices = (ROBOTS, OBJECTS, OBSTACLES, OBJECTIVES)
  labels = ("robot", "object", "obstacle", "objective")
  for label, value, valid in zip(labels, values, choices):
    if value not in valid:
      raise ValueError(
          f"Unknown {label} {value!r}. Valid choices: {', '.join(valid)}")


def task_from_sequence(values):
  if isinstance(values, Task):
    return values
  if isinstance(values, str):
    return parse_task_name(values)
  if len(values) != 4:
    raise ValueError(f"Task must have four components, got {values!r}")
  task = Task(*(normalize(x) for x in values))
  validate_task(task)
  return task


def all_combinations():
  for combo in itertools.product(ROBOTS, OBJECTS, OBSTACLES, OBJECTIVES):
    yield Task(*combo)


def discover(root):
  root = pathlib.Path(root)
  tasks = []
  if not root.exists():
    return tasks
  for child in root.iterdir():
    if not child.is_dir():
      continue
    data = child / "data.hdf5"
    if not data.exists():
      continue
    try:
      task = parse_task_name(child.name).with_path(data)
    except ValueError:
      continue
    tasks.append(task)
  return sorted(tasks)
