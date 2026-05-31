# Stage 2 Training Notes

Stage 2 trains System-2 alignment. It does not touch the diffusion policy.

## What Trains

- **Frozen**: Qwen2-VL base model weights.
- **Trainable**: LoRA adapters on `q_proj` and `v_proj` across all Qwen attention layers.
- **Trainable**: the `<SUBGOAL>` token embedding row inside Qwen.
- **Trainable**: MLP projector from Qwen hidden state → 512-d `policy_condition`.
- **Trainable**: auxiliary waypoint, gripper, and stage heads.
- **Not trained**: System 1 diffusion policy.

## Why LoRA

Without LoRA, the only trainable part of Qwen is a single token embedding row. The frozen
attention layers cannot re-route visual features toward the `<SUBGOAL>` position, so the
subgoal representation is severely limited regardless of how long you train.

LoRA adds low-rank adapters (r=8) to the query and value projections in every attention layer.
This gives the model capacity to learn which visual and language features are relevant to
predicting physical subgoals, without updating the full 2B/7B backbone.

Observed effect: typical non-spike loss drops from 0.5–2.0 (no LoRA) to 0.005–0.05 (LoRA r=8).

## Architecture Flow

```
image + task text + <SUBGOAL>
        |
   Qwen2-VL backbone
   (frozen weights + trainable LoRA on q_proj, v_proj)
        |
 z_subgoal  ← final hidden state at <SUBGOAL> token position
        |
   MLP projector  (trainable)
        |
 policy_condition  512-d
        |
 waypoint head + gripper head + stage head  (trainable)
        |
 auxiliary loss
```

The heads predict from `policy_condition`, not directly from `z_subgoal`. This forces
gradients to flow through the full path: aux heads → projector → `<SUBGOAL>` token → LoRA.

## One Training Step

```
1. Load from dataset:
   - camera image (agentview)
   - task language string
   - target_waypoint   (7-DOF end-effector pose)
   - target_gripper_state  (binary: open / closed)
   - target_stage_class    (3-class: approach / manipulation / retraction)

2. Build Qwen input:
   "Robot task: {task}. Predict the next physical subgoal. <SUBGOAL>"
   + camera image

3. Forward Qwen2-VL (frozen weights + LoRA):
   → z_subgoal  [B, hidden_size]

4. MLP projector:
   z_subgoal → policy_condition  [B, 512]

5. Auxiliary heads:
   policy_condition → predicted waypoint      (MSE vs target_waypoint)
   policy_condition → predicted gripper logit (BCE vs target_gripper_state)
   policy_condition → predicted stage logits  (CE  vs target_stage_class)

6. Backprop updates: LoRA adapters, <SUBGOAL> embedding, projector, heads.
```

## Loss Weights and Why They Changed

The total loss is:

```
loss = waypoint_weight * waypoint_loss
     + gripper_weight  * gripper_loss
     + stage_weight    * stage_loss
```

**Original weights (1.0 / 1.0 / 1.0) caused catastrophic oscillation.**

With `batch_size=2`, a batch can contain two samples with the same gripper state or the same
stage class. Binary cross-entropy and cross-entropy have no upper bound when the model is
confident-but-wrong on a skewed batch. This caused gripper and stage losses to spike to 4–6
every few steps, randomly yanking the projector in wrong directions.

Actual log evidence from the non-LoRA 100k run:
```
step=98825  gripper=4.648  (fine two steps later at 0.012)
step=99200  gripper=3.536
step=99475  stage=3.557
```

Two fixes applied together:
1. `batch_size 2 → 8`: each batch now contains ~2–3 samples per class, making the gradient
   estimate far more stable.
2. `gripper_weight 1.0 → 0.3`, `stage_weight 1.0 → 0.3`: caps the maximum damage from a
   pathological batch, keeping classification contribution on the same scale as waypoint MSE.

**waypoint_weight stays at 1.0** — waypoint MSE converges fast and is the physically
meaningful signal. It was at ~0.01 by step 100 in both runs.

## Run (v2 — current)

```bash
sbatch slurm_training/qwenlora/train_stage2_qwen_lora.slurm    # Qwen2-VL-2B
sbatch slurm_training/qwen7lora/train_stage2_qwen7_lora.slurm  # Qwen2-VL-7B
```

Current hyperparameters (v2):

| Parameter | Value | Reason |
|---|---|---|
| `batch_size` | 8 | stable gradient for classification losses |
| `lora_r` | 8 | gives attention layers capacity to route visual features |
| `lora_alpha` | 16 | standard 2× scaling |
| `lora_target_modules` | q_proj v_proj | attention input and value projections |
| `steps` | 200,000 | previous 100k run never fully converged |
| `warmup_steps` | 2,000 | 1% of total steps |
| `grad_clip` | 0.5 | tighter clip to further limit spike damage |
| `gripper_weight` | 0.3 | reduces classification spike impact |
| `stage_weight` | 0.3 | reduces classification spike impact |
| `lr_system2` | 5e-6 | conservative LR for LoRA + subgoal token |
| `lr_heads` | 1e-4 | faster LR for fresh MLP heads |

Output directories:

```
outputs_qwenlora_v2/stage2/final/system2_aux.pt   ← 2B
outputs_qwen7lora_v2/stage2/final/system2_aux.pt  ← 7B
```

## Loss — How To Judge Progress

Log line format:
```
[stage2/System2] step=... loss=... waypoint=... gripper=... stage=...
```

Expected behaviour with v2 config:

| Head | Starting value | Healthy end value |
|---|---|---|
| `waypoint_loss` | ~1.3 | <0.02 (converges fast, within first 1k steps) |
| `gripper_loss` | ~0.69 | <0.05 (below random-binary baseline) |
| `stage_loss` | ~1.10 | <0.10 (below random-3class baseline) |
| `loss` (total) | ~3.0 | <0.10 typical, occasional spikes <1.0 |

Warning signs:
- `waypoint` flat after 5k steps: LoRA or subgoal token not receiving gradient.
- `gripper` stuck near 0.69: class balance issue or LR too low for heads.
- `stage` stuck near 1.10: same.
- Spikes >3.0 persisting every few steps: batch_size still too small or grad_clip too high.
- `nan`: lower `--lr-system2` to `1e-6` or `--lr-heads` to `5e-5`.

## Relation to Stage 1 and Stage 3

Stage 2 is independent of Stage 1 — it can run in parallel. The only interface
constraint is:

```
MLP projector output dim (512) == System1Actor qwen_condition_dim (512)
```

Stage 3 loads both:
```
outputs/stage1_v3/stage1/final          ← System 1 diffusion policy
outputs_qwenlora_v2/stage2/final/       ← System 2 bundle (2B variant)
outputs_qwen7lora_v2/stage2/final/      ← System 2 bundle (7B variant)
```

The `system2_aux.pt` bundle contains: LoRA state dict, `<SUBGOAL>` token embedding,
projector weights, and auxiliary head weights.
