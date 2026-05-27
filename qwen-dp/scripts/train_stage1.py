#!/usr/bin/env python
"""Stage 1 / A0: train System 1 with zero System-2 conditioning."""

from __future__ import annotations

import argparse

import torch

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
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
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
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr_policy)
    scheduler = make_lr_scheduler(optimizer, args, args.steps)
    policy.train()

    for step, batch in enumerate(cycle(dataloader), start=1):
        batch = move_batch(batch, device)
        condition = zero_condition(batch, args.policy_condition_dim, device)
        loss, _ = policy(prepare_policy_batch(batch, condition))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0:
            print(f"[stage1/A0] step={step} policy_loss={loss.item():.6f}", flush=True)
        if step % args.save_every == 0:
            save_policy(policy, stage_dir / f"checkpoint_{step:06d}")
        if step >= args.steps:
            break

    save_policy(policy, stage_dir / "final")
    save_training_state(stage_dir, args.steps, {"policy_loss": float(loss.detach().cpu())})


if __name__ == "__main__":
    main()
