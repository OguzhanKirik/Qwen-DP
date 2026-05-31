# Training Slurm Scripts

The training scripts are grouped by model family:

- `qwen/`: Qwen2-VL-2B Stage1/Stage2/Stage3 scripts.
- `qwen7/`: Qwen2-VL-7B full fine-tuning scripts.
- `qwenlora/`: Qwen2-VL-2B LoRA scripts.
- `qwen7lora/`: Qwen2-VL-7B LoRA scripts.

Current async Stage3 scripts use `--async-condition-interval 4`, so System2 conditioning is refreshed every 4 environment steps during training.

## Common Commands

```bash
sbatch slurm_training/qwen/train_stage3_async_k4.slurm
sbatch slurm_training/qwen7/train_stage3_async_k4_qwen7.slurm
sbatch slurm_training/qwenlora/train_stage2_qwen_lora.slurm
sbatch slurm_training/qwen7lora/train_stage2_qwen7_lora.slurm
```

After LoRA Stage2 succeeds, submit:

```bash
sbatch slurm_training/qwenlora/train_stage3_async_k4_qwen_lora.slurm
sbatch slurm_training/qwen7lora/train_stage3_async_k4_qwen7_lora.slurm
```
