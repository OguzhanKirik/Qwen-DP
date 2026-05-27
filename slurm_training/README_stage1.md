# Stage 1 Training Notes

Stage 1 trains the A0 baseline: System 1 only, with zero System-2 conditioning.

## What Trains

- Trainable: the full `System1Actor` diffusion policy.
- Not used: Qwen/System 2.
- Conditioning: a zero vector is passed as `qwen_dp.policy_condition`.

This gives us the motor-control baseline for later ablations.

## Run

```bash
sbatch slurm_training/train_stage1.slurm
```

Logs are written to:

```text
slurm_training/logs/stage1_<jobid>.out
slurm_training/logs/stage1_<jobid>.err
```

## Loss

The log line looks like:

```text
[stage1/A0] step=... policy_loss=...
```

`policy_loss` is the diffusion policy training loss from LeRobot. In practical terms, the policy adds noise to expert action chunks and learns to predict the denoising target. Lower loss means the policy is better matching the expert action distribution under the current visual/state observations.

Read it as a training signal, not as direct task success. A lower diffusion loss usually helps, but the real test is rollout success rate and step count.

## Healthy Progress

- The loss should trend downward over hundreds or thousands of steps.
- Short-term noise is normal; do not judge from one printed line.
- If the loss is flat from the beginning, check data loading and whether observations/actions have sane values.
- If the loss becomes `nan`, lower the learning rate or gradient clip.
- If the job is slow but stable, prioritize finishing a checkpoint over chasing perfect loss.

## Hyperparameter Tuning

The Python defaults are intentionally used by `train_stage1.slurm`.

Important knobs if you override them manually:

```bash
STEPS=...
BATCH_SIZE=...
LR_POLICY=...
SAVE_EVERY=...
LOG_EVERY=...
```

Recommended first changes:

- More stable training: lower `LR_POLICY` from `1e-4` to `5e-5`.
- Out of memory: lower `BATCH_SIZE`.
- Too slow to verify: lower `STEPS` for a smoke test.
- Need more frequent checkpoints: lower `SAVE_EVERY`.

## What To Save

Final output goes to:

```text
outputs/stage1/final
```

Use this as the A0 baseline checkpoint and as the System 1 initialization for later joint training.
