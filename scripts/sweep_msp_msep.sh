#!/usr/bin/env bash
# Grid sweep of the MoSS sparsity (msp) and route-separation (msep) loss
# weights, full offline_comp schedule, a few trials in parallel on one GPU.
#
# LEGACY: written for the branch before the revision. With the revised
# defaults msep is the hinged environment-expert term and mcomp/mz are on;
# use scripts/sweep_router.sh for the revised objective, or append
# offline_comp_legacy to COMMON's --configs to reproduce the earlier runs.
#
#   bash scripts/sweep_msp_msep.sh                  # run the 9 trials
#   DRY_RUN=1 bash scripts/sweep_msp_msep.sh        # print the commands
#   PARALLEL=1 bash scripts/sweep_msp_msep.sh       # one trial at a time
#   MSP_LIST='1e-3' MSEP_LIST='0.3 1.0' bash ...     # a sub-grid
#   bash scripts/sweep_msp_msep.sh --run.steps 80   # extra flags go to every trial
#
# A trial whose logdir holds a `done` marker is skipped; an unfinished one
# resumes from its own checkpoint (logdir/ckpt). Summarize with
# `python scripts/sweep_summary.py`.
set -u

read -r -a MSP <<< "${MSP_LIST:-1e-4 1e-3 1e-2}"
read -r -a MSEP <<< "${MSEP_LIST:-0.3 1.0 3.0}"
PARALLEL=${PARALLEL:-2}
SWEEP_ROOT=${SWEEP_ROOT:-./logdir/sweep_msp_msep}
DRY_RUN=${DRY_RUN:-0}
GROUP=${GROUP:-sweep-msp-msep-e8k2}
PROJECT=${PROJECT:-offline-comp-moss}
EXTRA=("$@")

COMMON=(
  --configs offline_comp offline_comp_moss --seed 42
  --agent.dyn.moss.experts 8 --agent.dyn.moss.topk 2
  --agent.moss_scales.mbal 0
  --agent.offline.bc_scale 0.1
  --run.env_eval_every 50000
  --run.save_every 10000
  --logger.wandb_project "$PROJECT"
  --logger.wandb_group "$GROUP"
)

cd "$(dirname "$0")/.."
mkdir -p "$SWEEP_ROOT"

run_trial() {
  local p=$1 s=$2
  local name="msp${p}_msep${s}"
  local logdir="$SWEEP_ROOT/$name"
  local cmd=(python methods/MoSS/main.py "${COMMON[@]}"
    --agent.moss_scales.msp "$p" --agent.moss_scales.msep "$s"
    --logger.wandb_name "sweep-msp${p}-msep${s}-e8k2"
    --logdir "$logdir" "${EXTRA[@]}")
  if [ "$DRY_RUN" = 1 ]; then
    echo "${cmd[*]}"
    return 0
  fi
  if [ -f "$logdir/done" ]; then
    echo "[sweep] skip $name (done)"
    return 0
  fi
  echo "[sweep] start $name $(date '+%F %T')"
  "${cmd[@]}" >> "$logdir.log" 2>&1
  local code=$?
  if [ "$code" = 0 ]; then
    touch "$logdir/done"
    echo "[sweep] done  $name $(date '+%F %T')"
  else
    echo "[sweep] FAIL  $name $(date '+%F %T') (exit $code, see $logdir.log)"
  fi
}

for p in "${MSP[@]}"; do
  for s in "${MSEP[@]}"; do
    if [ "$DRY_RUN" != 1 ]; then
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
      run_trial "$p" "$s" &
      sleep 60  # stagger startup (dataset load, JAX compile)
    else
      run_trial "$p" "$s"
    fi
  done
done
wait
echo "[sweep] all trials finished $(date '+%F %T')"
