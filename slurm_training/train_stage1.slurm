#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen-dp-stage1
#SBATCH --output=slurm_training/logs/stage1_%j.out
#SBATCH --error=slurm_training/logs/stage1_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

set -euo pipefail

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment
mkdir -p slurm_training/logs

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"

# Run with default hyperparameters defined in Python scripts
"$PYTHON_BIN" -u qwen-dp/scripts/train_stage1.py --device cuda
