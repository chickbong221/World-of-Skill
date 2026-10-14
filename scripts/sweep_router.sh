#!/usr/bin/env bash
# Grid sweep of the revised MoSS router objective: the competitive router loss
# (mcomp) x the hinged environment-expert term (msep), full offline_comp
# schedule, a few trials in parallel on one GPU. Rank the trials on the
# validation split (held-out rows of the train tasks), never on eval/ (the
# held-out recombination tasks):  python scripts/sweep_summary.py --root ...
#
#   MASK=77:93 bash scripts/sweep_router.sh          # hide the task indicator
#   DRY_RUN=1 bash scripts/sweep_router.sh           # print the commands
#   MCOMP_LIST='0 0.1' MSEP_LIST='1.0' bash ...      # a sub-grid
#   bash scripts/sweep_router.sh --run.steps 80      # extra flags to every trial
#
# MASK is the --agent.mask_obs value printed by
#   python -m offline_comp.task_obs_dims --root data/expert-panda-offline-comp-data
# Leave it empty only for the "task indicator visible" ablation.
set -u

read -r -a MCOMP <<< "${MCOMP_LIST:-0.0 0.03 0.1 0.3}"
read -r -a MSEP <<< "${MSEP_LIST:-0.0 0.3 1.0}"
PARALLEL=${PARALLEL:-2}
SWEEP_ROOT=${SWEEP_ROOT:-./logdir/sweep_router}
DRY_RUN=${DRY_RUN:-0}
MASK=${MASK:-}
GROUP=${GROUP:-sweep-router-e8k2}
PROJECT=${PROJECT:-offline-comp-moss}
EXTRA=("$@")

if [ -z "$MASK" ]; then
  echo "[sweep] WARNING: MASK is empty, the task indicator in 'vector' is" \
       "visible to the model (run offline_comp.task_obs_dims)" >&2
fi

COMMON=(
  --configs offline_comp offline_comp_moss --seed 42
  --agent.dyn.moss.experts 8 --agent.dyn.moss.topk 2
  --run.env_eval_every 50000
  --run.save_every 10000
  --logger.wandb_project "$PROJECT"
  --logger.wandb_group "$GROUP"
)

if [ -n "$MASK" ]; then
  COMMON+=(--agent.mask_obs "$MASK")
fi

cd "$(dirname "$0")/.."
mkdir -p "$SWEEP_ROOT"

run_trial() {
  local c=$1 s=$2
  local name="mcomp${c}_msep${s}"
  local logdir="$SWEEP_ROOT/$name"
  local cmd=(python methods/MoSS/main.py "${COMMON[@]}"
    --agent.moss_scales.mcomp "$c" --agent.moss_scales.msep "$s"
    --logger.wandb_name "sweep-mcomp${c}-msep${s}-e8k2"
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

for c in "${MCOMP[@]}"; do
  for s in "${MSEP[@]}"; do
    if [ "$DRY_RUN" != 1 ]; then
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
      run_trial "$c" "$s" &
      sleep 60  # stagger startup (dataset load, JAX compile)
    else
      run_trial "$c" "$s"
    fi
  done
done
wait
echo "[sweep] all trials finished $(date '+%F %T')"
