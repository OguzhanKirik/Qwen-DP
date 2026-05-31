#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=qwen-dp-stage1-v4-600k
#SBATCH --output=slurm_training/logs/stage1_v4_resume_600k_%j.out
#SBATCH --error=slurm_training/logs/stage1_v4_resume_600k_%j.err
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

"$PYTHON_BIN" -u qwen-dp/scripts/train_stage1.py \
    --device cuda \
    --output-dir outputs/stage1_v4 \
    --batch-size 32 \
    --lr-policy 3e-5 \
    --steps 600000 \
    --resume-from outputs/stage1_v4/stage1/checkpoint_200000 \
    --resume-step 200000 \
    --num-workers 15
