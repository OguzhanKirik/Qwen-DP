#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --job-name=stage3-qwen7-noaux
#SBATCH --output=slurm_training/logs/stage3_qwen7_no_aux_%j.out
#SBATCH --error=slurm_training/logs/stage3_qwen7_no_aux_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_training/logs
export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"

python qwen-dp-7/scripts/train_stage3_no_aux.py \
    --dataset-root datasets/lerobot_libero_10_subgoals \
    --output-dir outputs_qwen7 \
    --model-path models/Qwen2-VL-7B \
    --stage1-checkpoint outputs/stage1/final \
    --stage2-checkpoint outputs_qwen7/stage2/final \
    --stage-name stage3_no_aux \
    --batch-size 2 \
    --steps 100000 \
    --save-every 2000 \
    --log-every 25 \
    --lr-policy 1e-4 \
    --lr-system2 5e-6 \
    --lr-heads 1e-4 \
    --joint-policy-weight 1.0
