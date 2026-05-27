# Stage 2 Training Notes

Stage 2 trains System-2 alignment. It does not train the diffusion policy.

## What Trains

- Frozen: Qwen2-VL base model.
- Trainable: only the `<SUBGOAL>` token embedding row inside Qwen.
- Trainable: the MLP projector from Qwen hidden state to policy condition.
- Trainable: auxiliary waypoint, gripper, and stage heads.
- Not trained: System 1 diffusion policy.

The goal is to make the Qwen `<SUBGOAL>` hidden state encode useful physical subgoal information before joint training.

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
 policy_condition, 512D
        |
 waypoint head + gripper head + stage head
        |
 auxiliary System-2 loss
```

The MLP projector is the bridge we eventually want System 1 to consume. The auxiliary heads intentionally predict from `policy_condition`, not directly from raw `z_subgoal`, so gradients train the whole Stage 2 path:

```text
auxiliary losses -> aux heads -> MLP projector -> <SUBGOAL> token
```

This matters because if the heads read directly from `z_subgoal`, the projector can exist in the code but receive no useful Stage 2 gradient.

## One Training Step

Stage 2 uses two different meanings of subgoal:

```text
<SUBGOAL> token  = learnable special token inserted into the Qwen prompt
target subgoals = labels created by preprocessing the LIBERO dataset
```

For one batch, the sequence is:

```text
1. Load from dataset:
   - camera image
   - task language
   - target_waypoint
   - target_gripper_state
   - target_stage_class

2. Build Qwen input:
   "Robot task: {task}. Predict the next physical subgoal. <SUBGOAL>"
   plus the camera image

3. Run frozen Qwen2-VL:
   image + task + <SUBGOAL> -> z_subgoal

4. Learn the Stage 2 representation:
   z_subgoal -> MLP projector -> policy_condition, 512D

5. Predict auxiliary labels from policy_condition:
   policy_condition -> waypoint head -> predicted waypoint
   policy_condition -> gripper head  -> predicted gripper state
   policy_condition -> stage head    -> predicted task stage

6. Compute loss against preprocessed labels:
   predicted waypoint vs target_waypoint
   predicted gripper  vs target_gripper_state
   predicted stage    vs target_stage_class

7. Backprop updates only:
   - <SUBGOAL> token embedding
   - MLP projector
   - auxiliary heads
```

The auxiliary heads do not directly control the robot. They are supervision tools that force `policy_condition` to contain useful physical information. Later, System 1 consumes this vector:

```text
image + robot state + policy_condition -> diffusion policy -> robot actions
```

## Relation To Stage 1

The assignment text says to freeze the baseline Diffusion Policy from Stage 1 during Stage 2. In the current implementation, Stage 2 does not instantiate or load the Stage 1 policy at all. Functionally, this is acceptable for latent pre-alignment because the Stage 2 loss only trains System 2 and the auxiliary heads.

The important interface match is:

```text
MLP projector output dim = policy_condition_dim = System 1 qwen_condition_dim
```

By default this is `512`. Later joint training should load both:

```text
outputs/stage1/final
outputs/stage2/final/system2_aux.pt
```

and connect the projected `policy_condition` into the diffusion policy.


## Run

```bash
sbatch slurm_training/train_stage2.slurm
```

Current Slurm overrides:

```bash
--batch-size 4
--steps 30000
--save-every 2000
--log-every 25
--lr-system2 5e-6
```

Everything else uses the Python defaults.

## Loss

The log line looks like:

```text
[stage2/System2] step=... loss=... waypoint=... gripper=... stage=...
```

The exact objective is:

```text
loss =
  waypoint_weight * waypoint_loss
+ gripper_weight  * gripper_loss
+ stage_weight    * stage_loss
```

With default weights, all three are `1.0`.

Components:

- `waypoint_loss`: MSE between the predicted future end-effector waypoint and `target_waypoint`.
- `gripper_loss`: binary cross entropy for open/closed gripper state.
- `stage_loss`: cross entropy for approach/manipulation/retraction stage.

## How To Judge Progress

- `loss` should trend down over hundreds or thousands of steps.
- `waypoint_loss` is the most important physical signal. It should clearly decrease over time.
- `gripper_loss` starts around `0.69` for random binary prediction. Below `0.69` means learning.
- `stage_loss` starts around `1.10` for random 3-class prediction. Below `1.10` means learning.
- Do not overreact to one noisy print line; compare windows of logs.

Good early signs:

```text
gripper < 0.69
stage < 1.10
waypoint steadily decreasing
```

Warning signs:

- `waypoint` flat: subgoal embedding is not learning spatial information.
- `gripper` flat near `0.69`: gripper labels may be noisy or the LR is too low for heads.
- `stage` flat near `1.10`: stage labels may be weak or class balance may be poor.
- `nan`: learning rate too high, bad batch values, or unstable mixed precision.

## Hyperparameter Tuning

Recommended default for one H100:

```bash
BATCH_SIZE=4
STEPS=30000
LR_SYSTEM2=5e-6
```

Useful overrides:

```bash
BATCH_SIZE=2 sbatch slurm_training/train_stage2.slurm
LR_SYSTEM2=1e-6 sbatch slurm_training/train_stage2.slurm
STEPS=10000 sbatch slurm_training/train_stage2.slurm
```

Tuning rules:

- Out of memory: lower `BATCH_SIZE` to `2`.
- NaNs or unstable loss: lower `LR_SYSTEM2` to `1e-6`.
- Loss is stable but too slow: try `LR_SYSTEM2=1e-5`.
- Need a quick smoke test: set `STEPS=1000`.
- Need more checkpoints: lower `SAVE_EVERY`.

If classification losses improve but waypoint does not, consider changing the Python loss weights later, for example lowering `WAYPOINT_WEIGHT` only if waypoint numerically dominates the total loss. Keep the Slurm script simple unless we intentionally change those defaults.

## What To Save

Final output goes to:

```text
outputs/stage2/final/system2_aux.pt
```

This bundle contains the learned `<SUBGOAL>` embedding and auxiliary heads. Later joint training should combine this with the Stage 1 System 1 checkpoint.
