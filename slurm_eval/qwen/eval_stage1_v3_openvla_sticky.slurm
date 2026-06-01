#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=eval-a0-v3-ovs
#SBATCH --output=slurm_eval/logs/eval_a0_v3_openvla_sticky_%j.out
#SBATCH --error=slurm_eval/logs/eval_a0_v3_openvla_sticky_%j.err
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=4

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_eval/logs
OUTPUT_DIR="slurm_eval/results_half/stage1_v3/a0_openvla_sticky"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

LATEST_CHECKPOINT="$(find outputs/stage1_v3/stage1 -maxdepth 1 -type d -name 'checkpoint_*' | sort -V | tail -n 1)"
if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "No checkpoint found under outputs/stage1_v3/stage1" >&2
    exit 1
fi

echo "========================================="
echo "Evaluating Stage1 v3 A0 @ latest checkpoint"
echo "Sticky gripper: fixed continuous OpenVLA hysteresis"
echo "Checkpoint: $LATEST_CHECKPOINT"
echo "Environment: LIBERO-10"
echo "Episodes: 5 (with 5 videos)"
echo "========================================="

python qwen-dp/scripts/eval_stage1_a0.py \
    --checkpoint="$LATEST_CHECKPOINT" \
    --env_type=libero \
    --env_task=libero_10 \
    --gripper-mode=openvla_sticky \
    --n_episodes=5 \
    --batch_size=5 \
    --max_videos=5 \
    --device=cuda \
    --output_dir="$OUTPUT_DIR" \
    --seed=1000

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Videos saved to: $OUTPUT_DIR/rollout_videos"
echo "========================================="
