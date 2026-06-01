#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --job-name=qwen7-dp-stage1-t0
#SBATCH --output=slurm_training/logs/stage1_t0_%j.out
#SBATCH --error=slurm_training/logs/stage1_t0_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

# Stage 1 — single-task (task 0: put white mug on left plate + yellow mug on right plate)
# 1M steps, zero conditioning (no task-embed since only one task).
# Outputs to outputs_t0/stage1/

set -euo pipefail

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment
mkdir -p slurm_training/logs

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -u qwen-dp-7/scripts/train_stage1.py \
    --device cuda \
    --task-id 0 \
    --no-task-embed \
    --steps 1000000 \
    --save-every 10000 \
    --output-dir outputs_t0
