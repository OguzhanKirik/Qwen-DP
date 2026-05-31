# Qwen-DP

Qwen-DP is a dual-system robot policy for LIBERO-10. System 1 is a diffusion
policy that predicts low-level actions. System 2 is a Qwen2-VL planner that
reads the current image and task text, emits a `<SUBGOAL>` representation, and
feeds a 512-d conditioning vector into the diffusion policy.

## Architecture

- **System 1:** LeRobot-style diffusion policy (`System1Actor`) trained on
  expert trajectories.
- **System 2:** Qwen2-VL 2B or 7B vision-language backbone with a learned
  `<SUBGOAL>` token.
- **Bridge heads:** MLP heads project the Qwen hidden state into policy
  conditioning and auxiliary waypoint, gripper, and stage predictions.
- **LoRA variants:** `qwenlora` and `qwen7lora` train LoRA adapters on Qwen
  attention projections (`q_proj`, `v_proj`) while keeping the base Qwen
  backbone frozen.

## Training Flow

1. **Stage 1: System 1 policy**
   Trains the diffusion policy with learned task-ID conditioning.
   Current v4 run uses `outputs/stage1_v4/stage1/checkpoint_200000` as the
   shared policy checkpoint for later LoRA Stage 3 runs.

2. **Stage 2: System 2 alignment**
   Trains the Qwen bridge, auxiliary heads, learned `<SUBGOAL>` token, and
   optional LoRA adapters. The diffusion policy is not updated here.

3. **Stage 3: joint fine-tuning**
   Loads Stage 1 and Stage 2 checkpoints, then jointly trains the diffusion
   policy, trainable System 2 parameters, and auxiliary heads. The base Qwen
   weights stay frozen. Async Stage 3 uses `--async-condition-interval 4`.

## Slurm Training

Scripts live under `slurm_training/`:

```bash
# Qwen2-VL-2B
sbatch slurm_training/qwen/train_stage1_v4_resume_600k.slurm
sbatch slurm_training/qwen/train_stage2.slurm
sbatch slurm_training/qwen/train_stage3_async_k4.slurm

# Qwen2-VL-7B
sbatch slurm_training/qwen7/train_stage2_qwen7.slurm
sbatch slurm_training/qwen7/train_stage3_async_k4_qwen7.slurm

# LoRA variants
sbatch slurm_training/qwenlora/train_stage2_qwen_lora.slurm
sbatch slurm_training/qwen7lora/train_stage2_qwen7_lora.slurm
sbatch slurm_training/qwenlora/train_stage3_v4_async_k4_qwen_lora_v2_run2.slurm
sbatch slurm_training/qwen7lora/train_stage3_v4_async_k4_qwen7_lora_v2_run2.slurm
```

The run2 LoRA Stage 3 jobs save to:

```text
outputs_qwenlora_v2/stage3_v4_async_k4_run2/
outputs_qwen7lora_v2/stage3_v4_async_k4_run2/
```

## Evaluation

Evaluation scripts live under `slurm_eval/`. Common variants:

- **A0/A1:** System 1 baselines.
- **A2/A3:** Dual-system ablations.
- **A4:** Full async dual-system model.
- **A5:** A4 with recovery behavior.

Example commands:

```bash
sbatch slurm_eval/qwen/eval_stage1_v4.slurm
sbatch slurm_eval/qwenlora/eval_qwenlora_a4.slurm
sbatch slurm_eval/qwen7lora/eval_qwen7lora_a4.slurm
```

Evaluation outputs are written under `slurm_eval/results_*`, with logs,
`eval_info.json`, and rollout videos where enabled.

## Setup

See `SETUP.md` for the conda environment, pip dependencies, LeRobot install,
model download commands, dataset layout, and smoke tests. Large artifacts are
kept out of git; see `datasets/README.md`, `models/README.md`, and
`lerobot/README_QWEN_DP.md` for placeholders.

## Key Paths

```text
qwen-dp/                  Qwen2-VL-2B code path
qwen-dp-7/                Qwen2-VL-7B code path
slurm_training/           Training jobs
slurm_eval/               Evaluation jobs
datasets/                 LIBERO datasets
models/Qwen2-VL-2B/       Local 2B checkpoint
models/Qwen2-VL-7B/       Local 7B checkpoint
```
