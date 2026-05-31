#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --job-name=qwen7-dp-stage1
#SBATCH --output=slurm_training/logs/stage1_qwen7_%j.out
#SBATCH --error=slurm_training/logs/stage1_qwen7_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

set -euo pipefail

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment
mkdir -p slurm_training/logs

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"

# Stage 1 is Qwen-agnostic — same diffusion policy regardless of 2B or 7B variant.
# batch_size=32, crop_size=128, lr=3e-5, steps=600k, task-ID embedding enabled by default.
# The resulting checkpoint can be shared with the qwen (2B) Stage 3 experiments.
"$PYTHON_BIN" -u qwen-dp-7/scripts/train_stage1.py \
    --device cuda \
    --output-dir outputs_qwen7/stage1_v3
