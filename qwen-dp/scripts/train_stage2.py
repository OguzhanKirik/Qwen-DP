#!/usr/bin/env python
"""Stage 2: train the System-2 subgoal token, projector, and auxiliary heads."""

from __future__ import annotations

import argparse

import torch

from train_common import (
    StageLossWeights,
    add_common_args,
    add_system2_args,
    aux_targets,
    cycle,
    iter_parameters,
    jsonable_args,
    make_aux_heads,
    make_dataloader,
    make_dataset,
    make_lr_scheduler,
    make_system2,
    module_dtype,
    move_batch,
    qwen_inputs,
    save_system2_bundle,
    save_training_state,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_system2_args(parser)
    parser.add_argument("--steps", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    stage_dir = args.output_dir / "stage2"
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(stage_dir / "config.json", jsonable_args(args))

    dataset = make_dataset(args)
    dataloader = make_dataloader(args, dataset)
    device = torch.device(args.device)

    planner = make_system2(args)
    heads = make_aux_heads(args, planner)
    weights = StageLossWeights(args.waypoint_weight, args.gripper_weight, args.stage_weight)
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for _, param in planner.trainable_parameters()], "lr": args.lr_system2},
            {"params": heads.parameters(), "lr": args.lr_heads},
        ]
    )
    scheduler = make_lr_scheduler(optimizer, args, args.steps)
    planner.train()
    heads.train()

    for step, batch in enumerate(cycle(dataloader), start=1):
        batch = move_batch(batch, device)
        inputs = qwen_inputs(planner, batch, args)
        z_subgoal = planner(**inputs).to(device=device, dtype=module_dtype(heads))
        outputs = heads(z_subgoal)
        losses = heads.losses(
            outputs,
            aux_targets(batch),
            waypoint_weight=weights.waypoint,
            gripper_weight=weights.gripper,
            stage_weight=weights.stage,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(iter_parameters([heads, planner]), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0:
            print(
                "[stage2/System2] "
                f"step={step} loss={losses['loss'].item():.6f} "
                f"waypoint={losses['waypoint_loss'].item():.6f} "
                f"gripper={losses['gripper_loss'].item():.6f} "
                f"stage={losses['stage_loss'].item():.6f}",
                flush=True,
            )
        if step % args.save_every == 0:
            save_system2_bundle(planner, heads, stage_dir / f"checkpoint_{step:06d}", step)
        if step >= args.steps:
            break

    save_system2_bundle(planner, heads, stage_dir / "final", args.steps)
    save_training_state(stage_dir, args.steps, {"system2_loss": float(losses["loss"].detach().cpu())})


if __name__ == "__main__":
    main()
