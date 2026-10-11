#!/bin/bash
# One offline CompoSuite run on H100-duong: offline_run.sh METHOD [main.py flags]
# METHOD: LEQ (DreamerV3 world model + LEQ), MoSS (MoSS world model + LEQ),
# C-LAP or TD-MPC2. Data, logdir (checkpoints) and wandb files live on
# /mnt/data/duongnm2; the logdir is keyed by job id, so a requeued job resumes.
METHOD=$1; shift
M=/mnt/data/duongnm2
case $METHOD in
  LEQ) RUN="methods/dreamerv3/main.py --configs offline_comp" ;;
  MoSS) RUN="methods/MoSS/main.py --configs offline_comp offline_comp_moss" ;;
  C-LAP) RUN="methods/clap/main.py --configs offline_comp" ;;
  TD-MPC2) RUN="methods/tdmpc2/main.py --configs offline_comp" ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac
echo "$(date) start $METHOD, job $SLURM_JOB_ID, commit $(git rev-parse --short HEAD)"
python $RUN \
  --logdir $M/logdir/world-of-skill/${METHOD}_$SLURM_JOB_ID \
  --data.root $M/data/expert-panda-offline-comp-data \
  --logger.wandb_project offline-comp-baselines --logger.wandb_name $METHOD \
  --run.env_eval_workers $((SLURM_CPUS_PER_TASK - 2)) "$@"
rc=$?
echo "$(date) $METHOD exit $rc"
exit $rc
