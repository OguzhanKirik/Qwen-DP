#!/usr/bin/env python
"""Stage 1: train System 1 with learned task-ID conditioning.

A learnable embedding (one vector per task) is injected into the 512-d
FiLM conditioning slot instead of zeros. This forces the U-Net FiLM blocks
to actively use the conditioning pathway during Stage 1, so Stage 3 can
replace the task embedding with the Qwen projection without fighting against
400k+ steps of zero-conditioning inertia.

Pass --no-task-embed to fall back to the original zero-conditioning baseline.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from train_common import (
    add_common_args,
    cycle,
    jsonable_args,
    make_dataloader,
    make_dataset,
    make_lr_scheduler,
    make_policy_config,
    move_batch,
    prepare_policy_batch,
    save_policy,
    save_training_state,
    write_json,
    zero_condition,
)
from qwen_dp import System1Actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--steps", type=int, default=600000)
    parser.add_argument("--lr-policy", type=float, default=5e-5)
    parser.add_argument("--num-tasks", type=int, default=10,
                        help="Number of distinct tasks; sets the task embedding table size.")
    parser.add_argument("--no-task-embed", action="store_true",
                        help="Disable task-ID embedding and use zero conditioning instead.")
    parser.set_defaults(
        warmup_steps=6000,
        save_every=5000,
        batch_size=96,
        crop_size=128,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    stage_dir = args.output_dir / "stage1"
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(stage_dir / "config.json", jsonable_args(args))

    dataset = make_dataset(args)
    dataloader = make_dataloader(args, dataset)
    device = torch.device(args.device)

    policy = System1Actor(make_policy_config(args, dataset)).to(device)

    use_task_embed = not args.no_task_embed
    task_embed: nn.Embedding | None = None
    all_params = list(policy.parameters())
    if use_task_embed:
        task_embed = nn.Embedding(args.num_tasks, args.policy_condition_dim).to(device)
        all_params = all_params + list(task_embed.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=args.lr_policy)
    scheduler = make_lr_scheduler(optimizer, args, args.steps)
    policy.train()
    if task_embed is not None:
        task_embed.train()

    for step, batch in enumerate(cycle(dataloader), start=1):
        batch = move_batch(batch, device)

        if use_task_embed:
            task_ids = batch["task_index"].long().view(-1)  # [B,1] or [B] → [B]
            condition = task_embed(task_ids)
        else:
            condition = zero_condition(batch, args.policy_condition_dim, device)

        loss, _ = policy(prepare_policy_batch(batch, condition))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0:
            cond_tag = "task_embed" if use_task_embed else "zero_cond"
            print(f"[stage1] step={step} policy_loss={loss.item():.6f} cond={cond_tag}", flush=True)
        if step % args.save_every == 0:
            ckpt_dir = stage_dir / f"checkpoint_{step:06d}"
            save_policy(policy, ckpt_dir)
            if task_embed is not None:
                torch.save(task_embed.state_dict(), ckpt_dir / "task_embed.pt")
        if step >= args.steps:
            break

    save_policy(policy, stage_dir / "final")
    if task_embed is not None:
        torch.save(task_embed.state_dict(), stage_dir / "final" / "task_embed.pt")
    save_training_state(stage_dir, args.steps, {"policy_loss": float(loss.detach().cpu())})


if __name__ == "__main__":
    main()
