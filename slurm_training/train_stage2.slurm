#!/usr/bin/env bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=stage2-qwen-dp
#SBATCH --output=slurm_training/logs/stage2_%j.out
#SBATCH --error=slurm_training/logs/stage2_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

set -euo pipefail

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment
mkdir -p slurm_training/logs outputs/stage2

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" qwen-dp/scripts/train_stage2.py \
  --device cuda \
  --batch-size "${BATCH_SIZE:-4}" \
  --steps "${STEPS:-30000}" \
  --save-every "${SAVE_EVERY:-2000}" \
  --log-every "${LOG_EVERY:-25}" \
  --lr-system2 "${LR_SYSTEM2:-5e-6}"
