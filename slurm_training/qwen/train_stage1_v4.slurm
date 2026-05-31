#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=qwen-dp-stage1-v4
#SBATCH --output=slurm_training/logs/stage1_v4_%j.out
#SBATCH --error=slurm_training/logs/stage1_v4_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --exclude=gx13v1

set -euo pipefail

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment
mkdir -p slurm_training/logs

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"

# batch_size=32, crop_size=128, lr=3e-5, steps=200k — fast eval checkpoint
"$PYTHON_BIN" -u qwen-dp/scripts/train_stage1.py \
    --device cuda \
    --output-dir outputs/stage1_v4 \
    --batch-size 32 \
    --lr-policy 3e-5 \
    --steps 200000 \
    --num-workers 15
