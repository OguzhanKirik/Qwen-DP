# Setup

This repo keeps code in git and expects large artifacts to be downloaded locally. The Slurm scripts assume a conda environment named `ur5`; `environment.yml` uses the same name for reproducibility.

## 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate ur5
```

## 2. Install LeRobot

The repo expects a local LeRobot checkout at `lerobot/`:

```bash
git clone https://github.com/huggingface/lerobot.git lerobot
pip install -e lerobot
```

The training and evaluation scripts set:

```bash
export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
```

Use `qwen-dp-7` instead of `qwen-dp` for the 7B variant.

## 3. Download Model Weights

Model weights are not tracked in git. Expected layout:

```text
models/
  Qwen2-VL-2B/
  Qwen2-VL-7B/
```

Download with:

```bash
huggingface-cli download Qwen/Qwen2-VL-2B --local-dir models/Qwen2-VL-2B
huggingface-cli download Qwen/Qwen2-VL-7B --local-dir models/Qwen2-VL-7B
```

## 4. Prepare Datasets

Datasets are not tracked in git. Expected layout:

```text
datasets/
  lerobot_libero_10/
  lerobot_libero_10_subgoals/
```

After placing the base LeRobot LIBERO-10 dataset at `datasets/lerobot_libero_10`, create the subgoal dataset:

```bash
python qwen-dp/scripts/preprocess_libero_subgoals.py \
  --input datasets/lerobot_libero_10 \
  --output datasets/lerobot_libero_10_subgoals
```

## 5. Smoke Tests

```bash
export PYTHONPATH="$PWD/qwen-dp:$PWD/lerobot/src:${PYTHONPATH:-}"
python qwen-dp/tests/check_qwen_dp_policy.py
python qwen-dp/tests/check_system2_qwen.py
```

For the 7B code path:

```bash
export PYTHONPATH="$PWD/qwen-dp-7:$PWD/lerobot/src:${PYTHONPATH:-}"
python qwen-dp-7/tests/check_qwen_dp_policy.py
python qwen-dp-7/tests/check_system2_qwen.py
```

## 6. Training and Evaluation

Training scripts are under `slurm_training/`; evaluation scripts are under `slurm_eval/`.

```bash
sbatch slurm_training/qwen/train_stage1_v4_resume_600k.slurm
sbatch slurm_training/qwenlora/train_stage3_v4_async_k4_qwen_lora_v2_run2.slurm
sbatch slurm_eval/qwenlora/eval_qwenlora_a4.slurm
```

For headless LIBERO/MuJoCo evaluation, the Slurm scripts set:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```
