#!/bin/bash
#SBATCH --job-name=wos-offline
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=/mnt/data/duongnm2/output/%x_%j.out
#SBATCH --error=/mnt/data/duongnm2/output/%x_%j.err

# Offline CompoSuite, LEQ policy phase (README "LEQ (default)"), on one H100
# MIG 3g.40gb slice of the H100-duong cluster:
#   sbatch -J wos-LEQ scripts/slurm_offline_comp.sh LEQ    # DreamerV3 world model
#   sbatch -J wos-MoSS scripts/slurm_offline_comp.sh MoSS  # MoSS world model
# Extra arguments go to main.py. Code, data, logdir (checkpoints), wandb files
# and these logs all live on /mnt/data/duongnm2. The logdir is keyed by the job
# id, so a requeued job resumes from its checkpoint and a new job starts fresh.
# Env `wos` = clone of `dreamer` + robosuite 1.4 stack ($M/envs/setup_wos.sh).

METHOD=${1:?usage: sbatch scripts/slurm_offline_comp.sh LEQ|MoSS [flags]}
M=/mnt/data/duongnm2
case $METHOD in
  LEQ) RUN="methods/dreamerv3/main.py --configs offline_comp" ;;
  MoSS) RUN="methods/MoSS/main.py --configs offline_comp offline_comp_moss" ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $M/envs/wos
cd $M/World-of-Skill
export WANDB_DIR=$M/wandb
echo "$(date) $METHOD job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES," \
  "commit $(git rev-parse --short HEAD)"

python $RUN \
  --logdir $M/logdir/world-of-skill/${METHOD}_$SLURM_JOB_ID \
  --data.root $M/data/expert-panda-offline-comp-data \
  --logger.wandb_project offline-comp-baselines --logger.wandb_name $METHOD \
  --run.env_eval_workers $((SLURM_CPUS_PER_TASK - 2)) "${@:2}"
rc=$?
echo "$(date) exit $rc"
exit $rc
