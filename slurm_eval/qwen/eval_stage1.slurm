#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=eval-a0
#SBATCH --output=slurm_eval/logs/eval_a0_%j.out
#SBATCH --error=slurm_eval/logs/eval_a0_%j.err
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Activate the environment
source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

# Ensure we are in the project root
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

# Create output directories
mkdir -p slurm_eval/logs
mkdir -p slurm_eval/results/stage1/a0

# Set Python path
export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"

# Set rendering backend for LIBERO/MuJoCo (use OSMesa for software rendering)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Run Model A0 evaluation: System 1 alone with zero conditioning
echo "========================================="
echo "Evaluating Model A0: System 1 Alone"
echo "Description: No conditioning (zero vector)"
echo "Stage 1 checkpoint: outputs/stage1/final"
echo "Environment: LIBERO-10"
echo "Episodes: 5 (with 5 videos)"
echo "========================================="

python qwen-dp/scripts/eval_stage1_a0.py \
    --checkpoint=outputs/stage1/final \
    --env_type=libero \
    --env_task=libero_10 \
    --n_episodes=5 \
    --batch_size=5 \
    --max_videos=5 \
    --device=cuda \
    --output_dir=slurm_eval/results/stage1/a0 \
    --seed=1000

echo ""
echo "========================================="
echo "Model A0 evaluation complete!"
echo "Results saved to: slurm_eval/results/stage1/a0/"
echo "========================================="
echo "Evaluation complete!"
echo "Results saved to: slurm_eval/results/stage1/a0"
echo "Videos saved to: slurm_eval/results/stage1/a0/rollout_videos"
echo "========================================="
