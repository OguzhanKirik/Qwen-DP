#!/bin/bash
#SBATCH -A aisc
#SBATCH -p aisc-batch
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100&ARCH:X86
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval-a3
#SBATCH --output=slurm_eval/logs/eval_a3_%j.out
#SBATCH --error=slurm_eval/logs/eval_a3_%j.err
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=4

set -euo pipefail

source /sc/home/oguzhan.kirik/conda3/etc/profile.d/conda.sh
conda activate ur5
cd /sc/projects/sci-aisc/oguzhan.kirik/sereact_assignment

mkdir -p slurm_eval/logs slurm_eval/results/stage3/a3
export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

python qwen-dp/scripts/eval_stage3_dual.py     --variant=a3     --policy-checkpoint=outputs/stage3_no_aux/final/policy     --system2-checkpoint=outputs/stage3_no_aux/final/system2     --model-path=models/Qwen2-VL-2B     --update-interval=4     --gripper-mode=openvla     --env_type=libero     --env_task=libero_10     --n_episodes=5     --batch_size=5     --max_videos=5     --device=cuda     --output_dir=slurm_eval/results/stage3/a3     --seed=1000
