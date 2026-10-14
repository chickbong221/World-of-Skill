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
python methods/dreamerv3/main.py --configs offline_comp
# MoSS:
python methods/MoSS/main.py --configs offline_comp offline_comp_moss
```

### Revised protocol (default)

`offline_comp` now follows the revised MoSS method draft; `CHANGES_REVISED.md`
lists every change and the flag that restores the old behaviour.

- **Hide the task indicator.** The 93-dim `vector` contains CompoSuite's task
  one-hots (`use_task_id_obs=True`). Find them and pass them to the model:

  ```bash
  python -m offline_comp.task_obs_dims --root data/expert-panda-offline-comp-data
  python methods/MoSS/main.py --configs offline_comp offline_comp_moss \
    --agent.mask_obs <dims printed above>
  ```

  Without `mask_obs` the run is the "task indicator visible" ablation. The
  loader also prints the detected dims at startup.
- **Labels are not decoded** (`agent.decode_labels: False`) and the DreamerV3
  baseline no longer feeds `task_id`/`task_axes` to its encoder
  (`agent.enc_labels: False`; live evaluation passed `task_id = 0`).
- **Validation split.** `data.val_frac: 0.1` holds out the last 10% of every
  train task's rows; reports on it are logged as `val/` every `run.val_every`.
  Select hyperparameters and checkpoints on `val/`, never on `eval/` (the
  held-out recombination tasks). `scripts/sweep_summary.py` ranks on `val/`.
- **BC regularization** `agent.offline.bc_scale: 0.1` for both methods.
- The matched baseline is `python methods/MoSS/main.py --configs offline_comp`
  (DreamerV3 RSSM in the same agent code, same input protocol).
- Reproduce the branch before the revision by stacking
  `offline_comp_legacy` last (both config files).

Overrides:

```bash
python methods/dreamerv3/main.py --configs offline_comp \
  --data.root data/expert-panda-offline-comp-data \
  --data.train.tasks Panda_Box_None_Push,Panda_Plate_ObjectWall_Shelf \
  --data.test.tasks Panda_Dumbbell_ObjectDoor_Trashcan
```

## Task Sampling

Default is mixed multitask batches. For sequential (one task per batch,
rotating every N batches):

```bash
python methods/dreamerv3/main.py --configs offline_comp \
  --data.sampling.schedule sequential \
  --data.sampling.batches_per_task 1000
```

Add `--data.sampling.shuffle_tasks true` to reshuffle order between passes.
`--data.sampling.eval_schedule sequential` applies the same to eval batches.

## Env Rollout Evaluation

Periodically rolls out the current policy in live CompoSuite envs and logs
`success_once`, `success_at_end`, `return` per task and averaged across the
split (`env_eval/train/*`, `env_eval/test/*`).

The policy is synced to the latest trained weights at the start of every
evaluation (`Agent.sync_policy`). Offline training never calls `policy()`
between updates, which is what refreshed the policy weights before, so earlier
evaluations ran on the initial random weights (first one) or on weights from
one evaluation earlier.

The envs run in worker processes (`embodied/run/env_eval.py`) and all of them
share one batched policy call per step. A round over the default 13 train + 3
test tasks x 10 episodes takes ~2 min (it took ~14 min with serial batch-1
rollouts); it is bounded by MuJoCo stepping (~6 ms per env step). The workers
only exist while an evaluation runs. Options (`--run.<name>`):

- `env_eval_workers`: env processes; `0` = half the cpus, `-1` = step the envs
  in the training process (no multiprocessing).
- `env_eval_slots`: parallel envs per task (default `1`). `2` is faster but
  doubles the RAM, each env needs ~0.7 GB.
- `env_eval_every`, `env_eval_episodes`, `env_eval_max_tasks` as before.

## Training Speed

The Dryad HDF5 files are gzip-compressed with column-wise chunks, so reading
one random 64-step window cost ~20 ms and the training loop spent 97-99% of
its time waiting for data. `offline_comp/dataset.py` now decompresses the
selected tasks once into RAM (~6 GiB for the default 13 train + 3 test tasks,
~30 s at startup) and gathers windows from memory, 0.15 ms instead of 650 ms
per batch of 32x64. The sampled batches are identical to the old reader's.

| per update, `offline_comp` (RTX 4090, shared) | before | after |
| --- | --- | --- |
| DreamerV3 | 586 ms | 53 ms |
| MoSS | 582 ms | 225 ms |

`perf/steps_per_sec` and `perf/data_wait_frac` (share of time waiting for a
batch) are logged next to the other metrics. The JAX profiler trace (~600 MB
per run) is now off by default, enable it with `--jax.profiler True`.

## Model-free Baselines (BC, TD3+BC, IQL)

`methods/offline_rl` ports the model-free baselines of
`cross_embodiment_offline_rl` to the same data, logger and live-env evaluation
as the Dreamer methods:

```bash
python methods/offline_rl/main.py --configs offline_comp bc
python methods/offline_rl/main.py --configs offline_comp td3bc
python methods/offline_rl/main.py --configs offline_comp iql
```

`offline_comp` selects the same 13 train / 3 held-out recombination split and
logs to wandb (`--logger.outputs jsonl` to keep it local). The losses and
update schedules follow the reference implementations: BC maximizes the
Gaussian log-likelihood; TD3+BC uses twin critics, target policy smoothing,
delayed actor and target updates and `-lambda Q + MSE` with
`lambda = alpha / mean|Q|`; IQL fits V with expectile regression, the twin
critics to `r + gamma V(s')` and the policy by advantage-weighted regression.
Differences to keep in mind:

- The networks are plain MLPs (the reference uses morphology-aware encoders
  for cross-embodiment locomotion), with a running observation normalizer.
- The recorded actions are a bounded expert action plus N(0, 1) noise that the
  env clips to [-1, 1]. BC, the Gaussian likelihoods and the TD3+BC regularizer
  therefore regress the raw recorded action (unbiased for the expert), while the
  critics see the executed, clipped action. Time-limit transitions are masked.
- `alpha` is the TD3+BC paper's 2.5 (reference: 0.01); IQL uses batch 1024 as
  in the reference. Everything is under `agent:` in
  `methods/offline_rl/configs.yaml`.
- One update consumes `batch_size` transitions and `run.steps` /
  `run.*_every` count transitions, e.g. 1M updates of TD3+BC is
  `--run.steps 2.56e8`.

## Offline DV2 Schedule (v-d4rl)

`--configs offline_comp` (MoSS: `offline_comp offline_comp_moss`), the default
offline setting, trains like v-d4rl's `offlinedv2/train_offline.py` instead of
jointly: the world model first, then the policy on the frozen model.

| | updates | batch x length | transitions |
| --- | --- | --- | --- |
| world model | 25,100 (v-d4rl `offline_model_train_steps` 25001, blocks of 100) | 64 x 50 | 80.3M |
| policy (actor + critic) | 200,000 (v-d4rl `steps`) | 64 x 50 | 640M |

- The losses are unchanged. The world-model phase minimizes the dynamics /
  representation, decoder, reward and continue terms (and MoSS's own); the policy
  phase minimizes the imagination actor-critic and replay-value terms from the
  posterior states of the frozen model (`agent_train` in v-d4rl). Each phase has
  its own optimizer (`opt_wm`, `opt_ac`), so the other phase's weights cannot
  move, and the model's usage statistics and router noise are off in the policy
  phase. Since the model is frozen there, the replay-value gradient no longer
  reaches the representation (`repval_grad`). Learning rates and sizes are kept.
- The logged `step` counts updates (`run.step_unit: updates`), 0 to 225,100:
  the world model first, then the policy. `offline/policy_updates`,
  `offline/updates` and `offline/transitions` are logged next to it.
- Live-env evaluation runs only in the policy phase, every 10K policy updates
  (v-d4rl evaluates 1 episode every 200 updates on a single task; 16 tasks x 10
  episodes cannot do that); the test-split report is also every 10K.
  Stack `offline_comp_joint` after `offline_comp` for the earlier joint protocol
  (1e9 transitions of 32 x 64, counted in transitions).
- `task_id` and `task_axes` are part of the world model loss of both agents (the
  decoder reconstructs them). MoSS still does not feed them to its encoder.

### MoSS routing mass per task

MoSS logs the mean soft router mass `pi` (softmax before top-k) that each task
sends to each expert, as `<prefix>/moss_route/<task>/e<m>` (the 8 values of a
task sum to 1):

- `train/`: the training batches, pooled over each log interval (sums and
  counts are averaged first and divided at log time). With the default
  `noise_impl: select` the logged mass is the clean softmax (the noise only
  changes which experts are selected); with `learned` it includes the noise.
- `report/`: a train-task batch in eval mode (no noise), every `report_every`.
- `eval/`: a held-out test-task batch in eval mode, every `eval_every`.

Keys use task names because the test split has its own `task_id`s. A task
missing from the window is not logged.

### MoSS router objective

The router is trained by the prediction losses through the mixture weights
and by three router-only terms (its input is detached):

- `mcomp`, competitive router loss (Jacobs et al., 1991):
  `-log sum_m pi~_m exp(-KL_m / tau)` over the active experts plus the same
  over all experts (`comp_dense`, free because all experts are computed). It
  pulls the router toward the per-transition posterior over experts, the
  signal the blended prediction loss loses once co-active experts agree.
- `msep`, environment-expert term. Default `sep_impl: hinge`:
  `[-H(E) + sum_g P(g) [H(E|g) - log s]_+] / log M` on the clean softmax
  masked to the noise-free active set, with `s = sep_target` (0 -> `topk`).
  `-H(E)` balances usage across tasks (replaces `mbal`); the hinge
  concentrates each task on about `s` experts and stops there. `sep_impl:
  plain` is the earlier `-I(task; expert) / min(log K, log M)` on the full
  soft mass, which prefers one expert per task.
- `mz`, router z-loss on the clean logits.

`msp`, `mbal` and `mdiv` remain for ablations (scale 0 by default). Sweep the
revised terms with `scripts/sweep_router.sh`.

Logged next to it: `moss_sep_mi` (nats), `moss_sep_mi_norm` (normalized by the
tasks present, so `eval/` is on the 3 test tasks' scale), `moss_route_HE` and
`moss_route_HEG` (all on the full soft mass), plus `moss_sep_HE_active` /
`moss_sep_HEG_active` for the hinge. Router health: `moss_eff_active`
(effective active experts per transition, near 1 = `k >= 2` collapsed),
`moss_select_agree` (training vs noise-free selection), `moss_noise_sigma`,
`moss_comp`, `moss_tau_route`, `moss_router_lse`; uncertainty threshold:
`moss_dbar` (world-model phase) and `moss_dbar_calib` (frozen model).

### Episode boundaries

A stored row is `(obs_t, a_t, r_t)` with the end flag of transition `t`, and the
obs after an episode's last row is not stored. That last row is therefore the
last record of its episode (`is_last`) and the next row, the reset observation of
the next episode, a first record (`is_first`, reward 0). Earlier the flags were
shifted by one row: the reset observation was flagged `is_last` with the previous
episode's last reward and no reset, so the model trained on a false transition
across episodes in the ~13% of windows that contain a boundary and the new
episode's first observation was never a first record. `test_episode_boundaries`
covers it; the model-free baselines now mask the transition into the next
episode (`is_first` of the next record), which drops the same transitions as
before.
