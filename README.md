# DreamerV3 Offline CompoSuite

Train DreamerV3 on the offline CompoSuite HDF5 datasets from Dryad
(https://datadryad.org/stash/dataset/doi:10.5061/dryad.9cnp5hqps), with
periodic live-env success evaluation.

Currently supports `expert-panda-offline-comp-data.tar.gz`.

## Setup

Python 3.11 on Linux or Mac.

```bash
pip install -U -r requirements.txt

git clone https://github.com/Lifelong-ML/CompoSuite.git
cd CompoSuite
pip install -r requirements_default.txt
pip install -e .
cd ..

pip install "mujoco==2.3.7" "numpy<2"
```

`robosuite==1.4.0` needs `mujoco 2.3.x` and `numpy 1.x`; newer versions break
its `mj_fullM` call.

## Data

Download `expert-panda-offline-comp-data.tar.gz` from the Dryad page above
(following guild in `https://neurotaxis.org/blog/2025/downloading_big_files_from_online_data_repositories.html` to get the link):

```bash
mkdir -p data
aria2c -c -x 8 -s 8 -k 1M -d data \
  -o expert-panda-offline-comp-data.tar.gz \
  'PASTE_S3_URL_HERE'

mkdir -p data/expert-panda-offline-comp-data
tar -xzf data/expert-panda-offline-comp-data.tar.gz \
  -C data/expert-panda-offline-comp-data
```

The extracted tree:

```text
data/expert-panda-offline-comp-data/
  Panda_<object>_<obstacle>_<objective>/data.hdf5
```

Components: `{IIWA, Jaco, Kinova3, Panda}` × `{Box, Dumbbell, Plate, Hollowbox}`
× `{None, GoalWall, ObjectDoor, ObjectWall}` × `{PickPlace, Push, Shelf, Trashcan}`.

Inspect discovered tasks:

```bash
python -m offline_comp.inspect --root data/expert-panda-offline-comp-data
```

## Train

```bash
python dreamerv3/main.py --configs offline_comp
```

Overrides:

```bash
python dreamerv3/main.py --configs offline_comp \
  --data.root data/expert-panda-offline-comp-data \
  --data.train.tasks Panda_Box_None_Push,Panda_Plate_ObjectWall_Shelf \
  --data.test.tasks Panda_Dumbbell_ObjectDoor_Trashcan
```

## Task Sampling

Default is mixed multitask batches. For sequential (one task per batch,
rotating every N batches):

```bash
python dreamerv3/main.py --configs offline_comp \
  --data.sampling.schedule sequential \
  --data.sampling.batches_per_task 1000
```

Add `--data.sampling.shuffle_tasks true` to reshuffle order between passes.
`--data.sampling.eval_schedule sequential` applies the same to eval batches.

## Env Rollout Evaluation

Periodically rolls out the current policy in live CompoSuite envs and logs
`success_once`, `success_at_end`, `return` per task and averaged across the
split (`env_eval/train/*`, `env_eval/test/*`). 

Each MuJoCo env takes 1–3 s to construct, so keep `env_eval_every` sparse.
