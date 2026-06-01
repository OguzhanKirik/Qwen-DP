#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eval-qwen7lora-a5
#SBATCH --output=slurm_eval/results_qwenlora7/logs/eval_qwen7lora_a5_%j.out
#SBATCH --error=slurm_eval/results_qwenlora7/logs/eval_qwen7lora_a5_%j.err

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

STAGE3_CHECKPOINT="${STAGE3_CHECKPOINT:-outputs_qwen7lora_v2/stage3_v4_async_k4_run3/checkpoint_088000}"
DATASET_ROOT="${DATASET_ROOT:-datasets/lerobot_libero_10_subgoals}"
CHECKPOINT_NAME="$(basename "$STAGE3_CHECKPOINT")"
BASE_RESULT_DIR="slurm_eval/results_qwenlora7/stage3_v4_async_k4_run3/$CHECKPOINT_NAME/a5_sequential"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
N_EPISODES="${N_EPISODES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-1}"

mkdir -p slurm_eval/results_qwenlora7/logs "$BASE_RESULT_DIR"
export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LEROBOT_LIBERO_DISABLE_STICKY_GRIPPER=1
export PYTHONUNBUFFERED=1

for TASK_ID in $TASK_IDS; do
    RESULT_DIR="$BASE_RESULT_DIR/task_${TASK_ID}"
    mkdir -p "$RESULT_DIR"
    echo "============================================================"
    echo "Running A5 task_id=$TASK_ID | episodes=$N_EPISODES | batch_size=$BATCH_SIZE"
    echo "Output: $RESULT_DIR"
    echo "============================================================"

    python -u qwen-dp-7/scripts/eval_stage3_dual.py \
        --variant=a5 \
        --policy-checkpoint="$STAGE3_CHECKPOINT/policy" \
        --system2-checkpoint="$STAGE3_CHECKPOINT/system2" \
        --dataset-root="$DATASET_ROOT" \
        --model-path=models/Qwen2-VL-7B \
        --system2-lora-r=8 \
        --system2-lora-alpha=16 \
        --system2-lora-dropout=0.05 \
        --system2-lora-target-modules q_proj v_proj \
        --update-interval=4 \
        --recovery-threshold=0.15 \
        --gripper-mode=native \
        --env_type=libero \
        --env_task=libero_10 \
        --task-ids "$TASK_ID" \
        --n_episodes="$N_EPISODES" \
        --batch_size="$BATCH_SIZE" \
        --max_videos="$MAX_VIDEOS" \
        --device=cuda \
        --output_dir="$RESULT_DIR" \
        --seed=1000

done
