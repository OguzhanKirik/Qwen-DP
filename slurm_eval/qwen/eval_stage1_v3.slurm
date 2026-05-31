#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=eval-a0-v3
#SBATCH --output=slurm_eval/logs/eval_a0_v3_%j.out
#SBATCH --error=slurm_eval/logs/eval_a0_v3_%j.err
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=4

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5

cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_eval/logs
mkdir -p slurm_eval/results/A0_v3

export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

LATEST_CHECKPOINT="$(find outputs/stage1_v3/stage1 -maxdepth 1 -type d -name 'checkpoint_*' | sort | tail -n 1)"
if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "No checkpoint found under outputs/stage1_v3/stage1" >&2
    exit 1
fi

echo "========================================="
echo "Evaluating Stage1 v3 A0 @ latest checkpoint"
echo "Checkpoint: $LATEST_CHECKPOINT"
echo "Environment: LIBERO-10"
echo "Episodes: 5 (with 5 videos)"
echo "========================================="

python qwen-dp/scripts/eval_stage1_a0.py \
    --checkpoint="$LATEST_CHECKPOINT" \
    --env_type=libero \
    --env_task=libero_10 \
    --n_episodes=5 \
    --batch_size=5 \
    --max_videos=5 \
    --device=cuda \
    --output_dir=slurm_eval/results/A0_v3 \
    --seed=1000

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "Results saved to: slurm_eval/results/A0_v3"
echo "Videos saved to: slurm_eval/results/A0_v3/rollout_videos"
echo "========================================="
