#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --job-name=stage2-qwen7-t0
#SBATCH --output=slurm_training/logs/stage2_t0_%j.out
#SBATCH --error=slurm_training/logs/stage2_t0_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

# Stage 2 — train Qwen2-VL-7B subgoal token + projector + aux heads
# on task 0 only. Run AFTER stage 1 is complete.

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_training/logs
export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"

python qwen-dp-7/scripts/train_stage2.py \
    --dataset-root datasets/lerobot_libero_10_subgoals \
    --output-dir outputs_t0 \
    --model-path models/Qwen2-VL-7B \
    --task-id 0 \
    --batch-size 2 \
    --steps 100000 \
    --save-every 2000 \
    --log-every 25 \
    --lr-system2 5e-6 \
    --lr-heads 1e-4
