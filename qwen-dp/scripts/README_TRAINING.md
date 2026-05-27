# Phase 4 Training Stages

This folder contains three separate scripts for the Phase 4 curriculum. Run them in order unless you already have checkpoints from earlier stages.

## Stage 1: System 1 Baseline

Script: `train_stage1.py`

Goal: train the standalone low-level Diffusion Policy baseline, also called **A0**.

This stage trains `System1Actor` on expert action trajectories while the System-2 subgoal conditioning vector is fixed to zeros. The result is a reusable motor-control baseline for later ablations.

Default output folder: `outputs/stage1/`

Final policy checkpoint: `outputs/stage1/final/`

```bash
python qwen-dp/scripts/train_stage1.py
```

## Stage 2: System 2 Alignment

Script: `train_stage2.py`

Goal: align the frozen Qwen System-2 representation with robot-relevant physical targets.

This stage keeps the Qwen2-VL backbone frozen and trains only the learnable `<SUBGOAL>` token, the MLP projector, and the auxiliary waypoint, gripper, and stage heads. It does not update the Diffusion Policy.

Default output folder: `outputs/stage2/`

Final System-2 bundle: `outputs/stage2/final/system2_aux.pt`

```bash
python qwen-dp/scripts/train_stage2.py
```

## Stage 3: Joint Fine-Tuning

Script: `train_stage3.py`

Goal: produce the full dual-system model, using the Stage 1 policy and Stage 2 aligned System-2 bridge.

This stage loads the A0 policy and the aligned System-2 bundle, then jointly trains the Diffusion Policy, the learnable `<SUBGOAL>` token, projector, and auxiliary heads. The Qwen2-VL backbone remains frozen.

Default output folder: `outputs/stage3/`

Final policy checkpoint: `outputs/stage3/final/policy/`

Final System-2 bundle: `outputs/stage3/final/system2/system2_aux.pt`

```bash
python qwen-dp/scripts/train_stage3.py
```

To use non-default checkpoints:

```bash
python qwen-dp/scripts/train_stage3.py \
  --stage1-checkpoint outputs/stage1/final \
  --stage2-checkpoint outputs/stage2/final
```
