#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=eval-a0-v4
#SBATCH --output=slurm_eval/logs/eval_a0_v4_%j.out
#SBATCH --error=slurm_eval/logs/eval_a0_v4_%j.err
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=4

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

CHECKPOINT="outputs/stage1_v4/stage1/checkpoint_565000"
DATASET_ROOT="datasets/lerobot_libero_10_subgoals"
OUTPUT_DIR="slurm_eval/results/stage1/a0_v4_565k_native_corrected_taskids"

mkdir -p slurm_eval/logs
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LEROBOT_LIBERO_DISABLE_STICKY_GRIPPER=1
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Evaluating Stage1 v4 A0 @ 565k steps with corrected task IDs"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset root: $DATASET_ROOT"
echo "Environment: LIBERO-10"
echo "Episodes: 5 (with 5 videos)"
echo "Output: $OUTPUT_DIR"
echo "========================================="

python -u qwen-dp/scripts/eval_stage1_a0.py \
    --checkpoint="$CHECKPOINT" \
    --dataset-root="$DATASET_ROOT" \
    --env_type=libero \
    --env_task=libero_10 \
    --n_episodes=5 \
    --batch_size=5 \
    --max_videos=5 \
    --device=cuda \
    --gripper-mode=native \
    --output_dir="$OUTPUT_DIR" \
    --seed=1000

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Videos saved to: $OUTPUT_DIR/rollout_videos"
echo "========================================="
