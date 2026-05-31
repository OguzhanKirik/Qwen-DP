# Phase 4 Training Stages

This folder contains three separate scripts for the Phase 4 curriculum. Run them in order unless you already have checkpoints from earlier stages.

## Stage 1: System 1 with Task-ID Conditioning

Script: `train_stage1.py`

Goal: train the standalone low-level Diffusion Policy with a learned task-ID embedding that primes the FiLM conditioning pathway for Stage 3.

This stage trains `System1Actor` on expert action trajectories. Instead of a zero conditioning vector, a small `nn.Embedding(10, 512)` lookup table is trained alongside the policy. The `task_index` field (0–9) is read directly from each batch — it is a pre-existing label in the LeRobot dataset, one integer per frame identifying which of the 10 LIBERO-10 tasks it belongs to. The corresponding 512-d embedding vector is injected into the `qwen_dp.policy_condition` FiLM slot.

By the end of Stage 1 the FiLM blocks have learned to actively use that 512-d slot. At Stage 3 the embedding table is discarded and the Qwen auxiliary-heads projector writes into the same slot — the FiLM blocks adapt much more readily than if they had been trained on zeros.

Default output folder: `outputs/stage1_v3/`

Final policy checkpoint: `outputs/stage1_v3/stage1/final/`

This checkpoint is Qwen-agnostic and is shared by both the qwen (2B) and qwen7 (7B) Stage 3 runs.

```bash
python qwen-dp/scripts/train_stage1.py
```

Pass `--no-task-embed` to disable the embedding and use zero conditioning instead (original behaviour).

## Stage 2: System 2 Alignment

Script: `train_stage2.py`

Goal: align the frozen Qwen System-2 representation with robot-relevant physical targets.

This stage keeps the Qwen2-VL backbone frozen and trains only the learnable `<SUBGOAL>` token, the MLP projector, and the auxiliary waypoint, gripper, and stage heads. It does not update the Diffusion Policy.

Default output folder: `outputs/stage2/`

Final System-2 bundle: `outputs/stage2/final/system2_aux.pt`

```bash
python qwen-dp/scripts/train_stage2.py
```

Optional Qwen LoRA adapters can be enabled for attention projections while keeping
the base backbone frozen:

```bash
python qwen-dp/scripts/train_stage2.py \
  --system2-lora-r 8 \
  --system2-lora-alpha 16 \
  --system2-lora-target-modules q_proj v_proj \
  --output-dir outputs/qwen2b_lora_r8
```

Use the same `--system2-lora-*` options when loading that Stage 2 checkpoint in
Stage 3.

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
