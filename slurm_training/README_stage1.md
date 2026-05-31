# Stage 1 Training Notes

Stage 1 trains the `System1Actor` diffusion policy with **task-ID conditioning** — a learned embedding per LIBERO-10 task that primes the U-Net FiLM blocks to actively use the conditioning slot before Stage 3 introduces Qwen.

## What Trains

- Trainable: the full `System1Actor` diffusion policy + a small `nn.Embedding(10, 512)` table.
- Not used: Qwen / System 2.
- Conditioning: a **learned 512-d task embedding** (one vector per task, indexed by `task_index` from the dataset) is passed as `qwen_dp.policy_condition`.

Pass `--no-task-embed` to fall back to zero conditioning (the original approach).

## Why Task-ID Embeddings Instead of Zeros

Training with zeros for 600k steps teaches the FiLM residual blocks to **ignore** the conditioning slot entirely. When Stage 3 then tries to inject a meaningful Qwen projection into that slot, the U-Net has to unlearn the suppression behaviour.

The task-ID embedding avoids this. `task_index` is an integer (0–9) already stored per frame in the LeRobot dataset — it is not computed or inferred, just read from `batch["task_index"]`. The `nn.Embedding` table is a 10-row lookup: each row is a 512-d vector that gets optimised alongside the policy. By the end of Stage 1 each vector encodes something about that task's motion pattern.

At Stage 3 the embedding table is discarded. The Qwen auxiliary-heads projector outputs a 512-d vector into the exact same FiLM slot. Because the FiLM blocks spent 600k steps learning to use that slot, they adapt to Qwen's richer signal much more readily.

The `task_embed.pt` file saved alongside the policy checkpoint is not needed by Stage 3 and can be ignored.

## Run

```bash
sbatch slurm_training/qwen/train_stage1.slurm
```

Logs are written to:

```text
slurm_training/logs/stage1_<jobid>.out
slurm_training/logs/stage1_<jobid>.err
```

Stage 1 is Qwen-agnostic. The checkpoint at `outputs/stage1_v3/stage1/final` is shared by both the qwen (2B) and qwen7 (7B) Stage 3 experiments.

## Loss

The log line looks like:

```text
[stage1] step=... policy_loss=... cond=task_embed
```

`policy_loss` is the diffusion denoising loss. Lower is better, but the real test is rollout success rate. Expect it to decrease steadily for the first ~100k steps then slow.

## Healthy Progress

- Loss should trend downward over hundreds or thousands of steps.
- Short-term noise is normal; do not judge from one printed line.
- If loss is flat from step 1, check data loading and that `task_index` values are in range 0–9.
- If loss becomes `nan`, lower `--lr-policy` or `--grad-clip`.

## Current Hyperparameters (v3)

| Parameter | Value | Reason |
|---|---|---|
| `batch_size` | 96 | ~10 examples per task per step on H100 |
| `crop_size` | 128 | enough scene context; 84 discarded too much |
| `lr_policy` | 5e-5 | scaled with batch size (sqrt rule from 3e-5 @ 32) |
| `warmup_steps` | 6000 | 1% of total steps |
| `steps` | 600000 | ~57 epochs; 18× more gradient signal than original 100k run |
| `num_tasks` | 10 | matches LIBERO-10 |

Override any default by passing it explicitly to the slurm script or Python command.

## What To Save

Final output goes to:

```text
outputs/stage1_v3/stage1/final/          ← policy weights (System1Actor)
outputs/stage1_v3/stage1/final/task_embed.pt  ← embedding table (not needed by Stage 3)
```

Use `outputs/stage1_v3/stage1/final` as the Stage 3 `--stage1-checkpoint`.
