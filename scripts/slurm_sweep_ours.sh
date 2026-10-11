#!/bin/bash
#SBATCH --job-name=wos-leq-moss
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=/mnt/data/duongnm2/output/%x_%j.out
#SBATCH --error=/mnt/data/duongnm2/output/%x_%j.err

# Sweep of 2 runs on one H100 MIG 3g.40gb slice, one after another: LEQ (the
# DreamerV3 world model) then MoSS, both with the LEQ policy phase.
#   sbatch scripts/slurm_sweep_ours.sh
# Deliberately no `set -e`: a run that dies must not cancel the next.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Sweep: LEQ, MoSS"
echo "================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/duongnm2/envs/wos
cd /mnt/data/duongnm2/World-of-Skill
export PYTHONUNBUFFERED=1 WANDB_DIR=/mnt/data/duongnm2/wandb
nvidia-smi
nvidia-smi -l 100 > /mnt/data/duongnm2/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

bash scripts/offline_run.sh LEQ
bash scripts/offline_run.sh MoSS

kill $GPU_MONITOR_PID
echo "Job finished"
