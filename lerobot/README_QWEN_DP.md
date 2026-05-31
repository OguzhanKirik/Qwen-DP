# LeRobot Dependency

The full LeRobot checkout is not tracked in this repository.

Expected local layout:

```text
lerobot/
  src/lerobot/
```

Clone or install the LeRobot version used by the experiment before training or evaluation:

```bash
git clone https://github.com/huggingface/lerobot.git lerobot
pip install -e lerobot
```

The Slurm scripts set `PYTHONPATH` to include `lerobot/src`, so a local clone at this path is the expected cluster setup.
