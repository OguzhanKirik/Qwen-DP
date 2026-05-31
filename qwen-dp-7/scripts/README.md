# LIBERO Subgoal Preprocessing

This directory contains offline preprocessing utilities for the dual-system LIBERO workflow.

## `preprocess_libero_subgoals.py`

This script augments a LeRobot-format LIBERO dataset with three auxiliary System-2 training targets:

1. `target_waypoint`
2. `target_gripper_state`
3. `target_stage_class`

The script is non-destructive: it reads a source dataset and writes a new augmented dataset directory.

Default command:

```bash
/sc/home/oguzhan.kirik/conda3/envs/ur5/bin/python scripts/preprocess_libero_subgoals.py \
  --input-dir datasets/lerobot_libero_10 \
  --output-dir datasets/lerobot_libero_10_subgoals \
  --overwrite
```

## Generated Auxiliary Targets

### `target_waypoint`

Shape: `[7]`

Meaning:

```text
[future_eef_x, future_eef_y, future_eef_z, future_rx, future_ry, future_rz, future_rw]
```

Generation rule:

For each frame `t`, the script looks ahead to frame `t + K` inside the same episode and copies:

```python
observation.state[t + K][:7]
```

The default horizon is:

```text
K = 20 frames
```

Since LIBERO-10 is `10 fps`, this is roughly a 2 second future subgoal. Near the end of an episode, the future index is clipped to the final frame, so no target crosses episode boundaries.

Condition:

Generated for every frame in every episode.

### `target_gripper_state`

Shape: scalar integer

Classes:

```text
0 = open_or_moving
1 = closed_or_clamping
```

Generation rule:

For each frame `t`, the script looks at the future gripper action:

```python
action[t + K][6]
```

Then it thresholds the value into a binary label.

Default condition:

```text
target_gripper_state = 1 if future_gripper_action >= 0.35
target_gripper_state = 0 otherwise
```

This default corresponds to:

```bash
--closed-sign positive
--gripper-threshold 0.35
```

Why positive by default:

In the downloaded `lerobot_libero_10` dataset, episodes typically start with gripper action `-1` and switch to `+1` around grasp-like events. So for this dataset, positive actions are treated as closed/clamping labels for the auxiliary classifier.

If a different LIBERO export uses the opposite convention, run:

```bash
--closed-sign negative
```

Condition:

Generated for every frame in every episode.

### `target_stage_class`

Shape: scalar integer

Classes:

```text
0 = approach
1 = manipulation
2 = retraction
```

This is a heuristic behavioral segmentation label.

#### Stage 0: Approach

Meaning:

The robot is moving toward the first grasp/contact or interaction point.

Default condition:

Stage 0 starts at frame `0` and ends at the first close/clamp-like gripper transition.

The script searches for the first transition:

```text
not closed -> closed
```

using the same gripper threshold as `target_gripper_state`.

By default, very early transitions before frame `5` are ignored:

```bash
--min-approach-frames 5
```

Fallback:

If no close transition is found, the approach stage ends at approximately `35%` of the episode.

#### Stage 1: Manipulation

Meaning:

The robot is carrying, placing, opening, closing, inserting, or otherwise executing the main task interaction.

Default condition:

Stage 1 starts at `approach_end` and continues until a release/retraction-like transition or terminal settling is detected.

The script first looks for the last transition:

```text
closed -> not closed
```

after the approach phase.

This is useful for pick-place tasks where release marks the end of manipulation.

#### Stage 2: Retraction

Meaning:

The robot is finishing the episode after the main interaction, usually by releasing, backing away, or settling near the final state.

Default condition:

Stage 2 begins at the detected `retraction_start`.

The script chooses the earlier plausible event between:

```text
last closed -> not closed gripper transition
terminal EEF settling near the final pose
```

Terminal settling is detected when the end-effector position stays within:

```text
0.025 meters
```

of its final position for:

```text
8 consecutive frames
```

These defaults are controlled by:

```bash
--terminal-radius 0.025
--terminal-window 8
```

Fallback:

If no retraction signal is found, Stage 2 starts at approximately `80%` of the episode.

## Safety Constraints

The script never lets a generated target cross an episode boundary.

The output dataset preserves the original structure:

```text
data/
meta/
videos/
README.md
```

Only the parquet files under `data/` are rewritten with the new columns. Metadata files are copied and then updated with feature definitions and statistics for the three new targets.

## Verified LIBERO-10 Defaults

For `datasets/lerobot_libero_10`, a dry run produced:

```text
Episodes: 379
Frames: 101469
FPS: 10
Horizon: 20
Closed sign: positive

approach_end mean = 62.4 frames
retraction_start mean = 218.4 frames
episode_len mean = 267.7 frames
```

First five inferred cuts:

```text
ep=000 len=214 approach_end=40  retraction_start=200
ep=001 len=284 approach_end=45  retraction_start=234
ep=002 len=345 approach_end=100 retraction_start=228
ep=003 len=285 approach_end=95  retraction_start=164
ep=004 len=278 approach_end=48  retraction_start=275
```

## Important Caveat

`target_stage_class` is not ground truth from the simulator. It is a heuristic training signal derived from robot state and gripper actions. It is intended to give System 2 a useful progress signal for pre-alignment and ablation, not to perfectly annotate every semantic phase of every long-horizon task.
