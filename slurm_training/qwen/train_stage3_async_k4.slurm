#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=stage3-async-k4
#SBATCH --output=slurm_training/logs/stage3_async_k4_%j.out
#SBATCH --error=slurm_training/logs/stage3_async_k4_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1


# Activate the environment
source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

# Ensure we are in the project root
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

# Run Stage 3: Joint Fine-tuning (System 1 + System 2)
# Note: This requires completed checkpoints from stage1 and stage2
python qwen-dp/scripts/train_stage3.py \
    --dataset-root datasets/lerobot_libero_10_subgoals \
    --stage1-checkpoint outputs/stage1/final \
    --stage2-checkpoint outputs/stage2_v2/stage2/final \
    --stage-name stage3_async_k4 \
    --async-condition-interval 4 \
    --batch-size 2 \
    --steps 100000 \
    --save-every 2000 \
    --log-every 25 \
    --lr-policy 1e-4 \
    --lr-system2 5e-6 \
    --lr-heads 1e-4 \
    --joint-policy-weight 1.0 \
    --joint-aux-weight 1.0
