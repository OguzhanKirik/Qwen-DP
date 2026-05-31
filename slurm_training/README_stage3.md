# Stage 3 Training Notes

Stage 3 jointly fine-tunes System 1 (diffusion policy) and System 2 (Qwen-based planner) together.

## Prerequisites

- **Stage 1 checkpoint**: `outputs/stage1/final` (System 1 policy trained)
- **Stage 2 checkpoint**: `outputs/stage2/final` (System 2 alignment trained)

## What Trains

- Trainable: System 1 diffusion policy (all parameters)
- Trainable: System 2 `<SUBGOAL>` token embedding in Qwen
- Trainable: MLP projector from Qwen to policy condition
- Trainable: Auxiliary waypoint, gripper, and stage heads
- Frozen: Qwen2-VL base model (backbone remains frozen)

## Architecture Flow

```text
image + task text + <SUBGOAL>
        |
   frozen Qwen2-VL backbone
        |
 z_subgoal hidden state
        |
   MLP projector
        |
 policy_condition (512D)
        |
 System 1 Diffusion Policy → actions
        |
 combined policy + auxiliary loss
```

## Training Strategy

Stage 3 uses a joint loss combining:
- Policy loss (action prediction via diffusion)
- Auxiliary System-2 losses (waypoint, gripper state, stage classification)

This allows the System 2 planner to learn to provide better conditioning for System 1.

## Submitting the Job

**IMPORTANT**: Only submit Stage 3 after Stage 2 completes successfully!

```bash
# Check that stage 2 has finished and created the final checkpoint
ls outputs/stage2/final/

# Submit stage 3
sbatch slurm_training/qwen/train_stage3_async_k4.slurm
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Monitor training logs
tail -f slurm_training/logs/stage3_<JOBID>.out

# Check for errors
tail -f slurm_training/logs/stage3_<JOBID>.err
```

## Output

Checkpoints will be saved to:
- `outputs/stage3_async_k4/checkpoint_002000/`
- `outputs/stage3_async_k4/checkpoint_004000/`
- ...
- `outputs/stage3_async_k4/final/`
