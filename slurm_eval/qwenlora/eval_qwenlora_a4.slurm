#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eval-qwenlora-a4
#SBATCH --output=slurm_eval/results_qwenlora/logs/eval_qwenlora_a4_%j.out
#SBATCH --error=slurm_eval/results_qwenlora/logs/eval_qwenlora_a4_%j.err

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_eval/results_qwenlora/logs slurm_eval/results_qwenlora/stage3_v4_async_k4/a4
export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

python qwen-dp/scripts/eval_stage3_dual.py \
    --variant=a4 \
    --policy-checkpoint=outputs_qwenlora_v2/stage3_v4_async_k4/final/policy \
    --system2-checkpoint=outputs_qwenlora_v2/stage3_v4_async_k4/final/system2 \
    --model-path=models/Qwen2-VL-2B \
    --system2-lora-r=8 \
    --system2-lora-alpha=16 \
    --system2-lora-dropout=0.05 \
    --system2-lora-target-modules q_proj v_proj \
    --update-interval=4 \
    --gripper-mode=openvla \
    --env_type=libero \
    --env_task=libero_10 \
    --n_episodes=5 \
    --batch_size=5 \
    --max_videos=5 \
    --device=cuda \
    --output_dir=slurm_eval/results_qwenlora/stage3_v4_async_k4/a4 \
    --seed=1000
