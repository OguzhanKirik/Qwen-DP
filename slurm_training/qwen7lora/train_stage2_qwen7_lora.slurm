#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=stage2-qwen7-lora
#SBATCH --output=slurm_training/qwen7lora/logs/stage2_qwen7_lora_%j.out
#SBATCH --error=slurm_training/qwen7lora/logs/stage2_qwen7_lora_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gx13v1

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_training/qwen7lora/logs
export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"

python qwen-dp-7/scripts/train_stage2.py \
  --dataset-root datasets/lerobot_libero_10_subgoals \
  --output-dir outputs_qwen7lora_v2 \
  --model-path models/Qwen2-VL-7B \
  --system2-lora-r 8 \
  --system2-lora-alpha 16 \
  --system2-lora-dropout 0.05 \
  --system2-lora-target-modules q_proj v_proj \
  --batch-size 8 \
  --steps 200000 \
  --warmup-steps 2000 \
  --grad-clip 0.5 \
  --gripper-weight 0.3 \
  --stage-weight 0.3 \
  --save-every 5000 \
  --log-every 25 \
  --lr-system2 5e-6 \
  --lr-heads 1e-4
