#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=stage2-qwen-db
#SBATCH --output=slurm_training/logs/stage2_%j.out
#SBATCH --error=slurm_training/logs/stage2_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1


# Activate the environment
source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

# Ensure we are in the project root
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

# Run Stage 2: System 2 Alignment
python qwen-dp/scripts/train_stage2.py \
    --dataset-root datasets/lerobot_libero_10_subgoals \
    --batch-size 2 \
    --steps 100000 \
    --output-dir outputs/stage2_v2 \
    --save-every 2000 \
    --log-every 25 \
    --lr-system2 5e-6 \
    --lr-heads 1e-4