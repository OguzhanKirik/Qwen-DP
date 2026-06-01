#!/usr/bin/env python
"""Stage 2: train the System-2 subgoal token, projector, and auxiliary heads."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_common import (
    StageLossWeights,
    add_common_args,
    add_system2_args,
    aux_targets,
    cycle,
    iter_parameters,
    jsonable_args,
    load_system2_bundle,
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
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Warm-start from a Stage 2 checkpoint directory.")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Global step to continue from when using --resume-from.")
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

    planner = make_system2(args).to(device)
    heads = make_aux_heads(args, planner).to(device)
    if args.resume_from is not None:
        load_system2_bundle(planner, heads, Path(args.resume_from))
    weights = StageLossWeights(
        waypoint=args.waypoint_weight,
        gripper=args.gripper_weight,
        stage=args.stage_weight,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for _, param in planner.trainable_parameters()], "lr": args.lr_system2},
            {"params": heads.parameters(), "lr": args.lr_heads},
        ]
    )
    scheduler = make_lr_scheduler(optimizer, args, args.steps)
    start_step = int(args.resume_step)
    if scheduler is not None and start_step > 0:
        scheduler.step(start_step)
    planner.train()
    heads.train()

    # Pre-fetch task descriptions to avoid repeated table lookups.
    # When episodes are filtered, iterate only over the selected subset and
    # map absolute frame indices to relative indices in the filtered hf_dataset.
    task_descriptions = dataset.meta.tasks.index.tolist()
    episode_indices = dataset.episodes if dataset.episodes is not None else list(range(dataset.meta.total_episodes))
    abs_to_rel = getattr(dataset, "_absolute_to_relative_idx", None)
    max_ep = max(episode_indices)
    episodes_tasks: list[str] = [""] * (max_ep + 1)
    for ep_idx in episode_indices:
        first_abs = int(dataset.meta.episodes[ep_idx]['dataset_from_index'])
        first_rel = abs_to_rel[first_abs] if abs_to_rel is not None else first_abs
        task_idx = dataset.hf_dataset[first_rel]['task_index']
        episodes_tasks[ep_idx] = task_descriptions[task_idx]

    for step, batch in enumerate(cycle(dataloader), start=start_step + 1):
        # Inject task descriptions from metadata into the batch
        ep_indices = batch["episode_index"].tolist()
        batch["task"] = [episodes_tasks[i] for i in ep_indices]

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
