# Evaluation Scripts for Dual-System Architecture

This directory contains SLURM scripts for evaluating the trained models on the LIBERO benchmark, following the ablation study design from TaskSteps.md Phase 5.

## Directory Structure

```
slurm_eval/
├── README.md                 # This file
├── eval_stage1.slurm         # Evaluate Model A0 (System 1 alone)
├── logs/                     # SLURM output and error logs
└── results/                  # Evaluation results and videos
    ├── stage1/               # Stage 1 evaluation results
    ├── stage2/               # Stage 2 evaluation results (future)
    └── stage3/               # Stage 3 evaluation results (future)
```

## Model Variants for Ablation Study

Following TaskSteps.md Phase 5, Step 11:

| Model | Description | Script | Status |
|-------|-------------|--------|--------|
| **A0** | System 1 alone (no System 2) | `eval_stage1.slurm` | ✅ Ready |
| **A1** | System 1 + frozen text embedding | TBD | ⏳ Future |
| **A2** | System 1 + frozen static subgoal (first frame only) | TBD | ⏳ Future |
| **A3** | Full Dual System without auxiliary tasks | TBD | ⏳ Future |
| **A4** | Full Dual System with auxiliary tasks (async) | TBD | ⏳ Future |
| **A5** | A4 + active recovery trigger | TBD | ⏳ Future |

## Running Evaluations

### Prerequisites

Before running evaluations, ensure the corresponding training stage has completed:

```bash
# For Model A0, check that Stage 1 training is complete
ls -lh outputs/stage1/final/

# Should see:
# - config.json
# - model.safetensors
# - preprocessor_config.json
```

### Model A0: System 1 Baseline

Evaluate the diffusion policy trained in Stage 1 without any System 2 conditioning:

```bash
sbatch slurm_eval/eval_stage1.slurm
```

**Configuration:**
- **Checkpoint**: `outputs/stage1/final`
- **Episodes**: 50 rollouts per task
- **Batch size**: 10 parallel environments
- **Environment**: LIBERO-10 (10 manipulation tasks)

### Monitoring Evaluation

```bash
# Check job status
squeue -u $USER

# Monitor progress
tail -f slurm_eval/logs/eval_stage1_<JOBID>.out

# Check for errors
tail -f slurm_eval/logs/eval_stage1_<JOBID>.err
```

### Results

Evaluation results are saved to:
- `slurm_eval/results/stage1/a0/eval_info.json` - Success rates and metrics
- `slurm_eval/results/stage1/a0/rollout_videos/` - Recorded rollout videos

**Key Metrics:**
- **Success Rate**: Percentage of successful task completions
- **Average Steps**: Number of steps taken per episode
- **Per-Task Performance**: Success rate breakdown by LIBERO task

## Future Evaluations

### Stage 2 Evaluation (After Stage 2 Training Completes)

Stage 2 trains System 2 alignment components but doesn't have a full policy yet. Evaluation will focus on:
- Auxiliary head predictions (waypoint, gripper, stage)
- Subgoal representation quality

### Stage 3 Evaluation (After Stage 3 Training Completes)

Stage 3 enables full dual-system evaluation with multiple variants:
- **Model A4**: Full system with asynchronous System 2 updates
- **Model A5**: A4 with active recovery triggers
- **Error Recovery Tests**: Perturbation experiments (Phase 5, Step 12)
- **Bottleneck Analysis**: Reasoning vs execution failure breakdown (Phase 5, Step 13)

## Evaluation Configuration

The evaluation uses the LeRobot evaluation pipeline. Key parameters:

```python
--policy.path=outputs/stage1/final    # Path to trained model checkpoint
--env.type=libero                      # LIBERO simulation environment
--env.task=libero_10                   # 10-task benchmark
--eval.n_episodes=50                   # Number of evaluation rollouts
--eval.batch_size=10                   # Parallel environments
--policy.device=cuda                   # GPU evaluation
--output_dir=slurm_eval/results/...   # Where to save results
```

## Tips

1. **Parallel Evaluation**: Adjust `--eval.batch_size` based on GPU memory
2. **Episode Count**: More episodes (50+) give better statistical significance
3. **Video Recording**: Videos are automatically saved for analysis
4. **Deterministic Results**: Use `--seed` for reproducibility

## Ablation Study Analysis

After collecting results from all model variants, compare:

1. **Success Rate Comparison**: Does System 2 improve over baseline (A0)?
2. **Step Efficiency**: Does System 2 reduce the number of steps needed?
3. **Task-Specific Performance**: Which tasks benefit most from System 2?
4. **Error Recovery**: How does the system handle perturbations?

Results will be summarized in the final research report.
