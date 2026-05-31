# Datasets

Dataset files are not tracked in git because they are too large.

Expected local layout:

```text
datasets/
  lerobot_libero_10/
  lerobot_libero_10_subgoals/
```

## Download LIBERO-10

The base dataset should be a LeRobot-format LIBERO-10 dataset. Download it from Hugging Face with:

```bash
huggingface-cli download lerobot/libero_10 \
  --repo-type dataset \
  --local-dir datasets/lerobot_libero_10
```

If the dataset is hosted in a project-specific artifact store instead, place or symlink it to the same path:

```text
datasets/lerobot_libero_10/
```

## Create the Subgoal Dataset

`lerobot_libero_10_subgoals/` is the preprocessed version with subgoal annotations used by the dual-system training scripts. Create it with:

```bash
python qwen-dp/scripts/preprocess_libero_subgoals.py \
  --input datasets/lerobot_libero_10 \
  --output datasets/lerobot_libero_10_subgoals
```

The training scripts expect `datasets/lerobot_libero_10_subgoals` by default.
