# Models

Model weights are not tracked in git because they are too large.

Expected local layout:

```text
models/
  Qwen2-VL-2B/
  Qwen2-VL-7B/
```

Download the Qwen2-VL checkpoints into those folders before running Stage 2, Stage 3, or dual-system evaluation:

```bash
huggingface-cli download Qwen/Qwen2-VL-2B --local-dir models/Qwen2-VL-2B
huggingface-cli download Qwen/Qwen2-VL-7B --local-dir models/Qwen2-VL-7B
```

If you use a different model revision, keep the same folder names or update the `--model-path` arguments in the Slurm scripts.
