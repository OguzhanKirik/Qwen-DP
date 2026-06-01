#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=stage3-v4-qwen7-lorav2-k4-r4b4
#SBATCH --output=slurm_training/qwen7lora/logs/stage3_v4_qwen7_lorav2_async_k4_run4_bs4_%j.out
#SBATCH --error=slurm_training/qwen7lora/logs/stage3_v4_qwen7_lorav2_async_k4_run4_bs4_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_training/qwen7lora/logs
export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-outputs/stage1_v4/stage1/checkpoint_565000}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-outputs_qwen7lora_v2/stage2/checkpoint_295000}"

"$PYTHON_BIN" -u qwen-dp-7/scripts/train_stage3.py \
  --dataset-root datasets/lerobot_libero_10_subgoals \
  --output-dir outputs_qwen7lora_v2 \
  --model-path models/Qwen2-VL-7B \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --stage2-checkpoint "$STAGE2_CHECKPOINT" \
  --stage-name stage3_v4_async_k4_run4_bs4 \
  --async-condition-interval 4 \
  --system2-lora-r 8 \
  --system2-lora-alpha 16 \
  --system2-lora-dropout 0.05 \
  --system2-lora-target-modules q_proj v_proj \
  --crop-size 128 \
  --batch-size 4 \
  --steps 100000 \
  --save-every 1000 \
  --log-every 25 \
  --lr-policy 5e-5 \
  --lr-system2 5e-6 \
  --lr-heads 1e-4 \
  --joint-policy-weight 1.0 \
  --joint-aux-weight 1.0
