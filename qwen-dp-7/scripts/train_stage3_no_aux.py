#!/usr/bin/env python
"""Stage 3 / A3: jointly train System 1 and System 2 with auxiliary losses disabled."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_common import (
    StageLossWeights,
    add_async_condition_args,
    add_common_args,
    add_system2_args,
    async_condition_batch,
    aux_targets,
    cycle,
    episode_task_descriptions,
    iter_parameters,
    jsonable_args,
    load_system2_bundle,
    make_aux_heads,
    make_dataloader,
    make_dataset,
    make_lr_scheduler,
    make_policy_config,
    make_system2,
    module_dtype,
    move_batch,
    prepare_policy_batch,
    qwen_inputs,
    save_policy,
    save_system2_bundle,
    save_training_state,
    write_json,
)
from qwen_dp import System1Actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_system2_args(parser)
    add_async_condition_args(parser)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--joint-policy-weight", type=float, default=1.0)
    parser.add_argument("--stage1-checkpoint", type=str, default="outputs/stage1/final")
    parser.add_argument("--stage2-checkpoint", type=str, default="outputs_qwen7/stage2/final")
    parser.add_argument("--stage-name", type=str, default="stage3_no_aux")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    stage_dir = args.output_dir / args.stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    config = jsonable_args(args)
    config["joint_aux_weight"] = 0.0
    write_json(stage_dir / "config.json", config)

    dataset = make_dataset(args)
    dataloader = make_dataloader(args, dataset)
    device = torch.device(args.device)

    policy = System1Actor.from_pretrained(
        Path(args.stage1_checkpoint),
        config=make_policy_config(args, dataset),
    ).to(device)
    planner = make_system2(args).to(device)
    heads = make_aux_heads(args, planner).to(device)
    load_system2_bundle(planner, heads, Path(args.stage2_checkpoint))

    weights = StageLossWeights(
        waypoint=args.waypoint_weight,
        gripper=args.gripper_weight,
        stage=args.stage_weight,
        policy=args.joint_policy_weight,
        aux=0.0,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": policy.parameters(), "lr": args.lr_policy},
            {"params": [param for _, param in planner.trainable_parameters()], "lr": args.lr_system2},
            {"params": heads.parameters(), "lr": args.lr_heads},
        ]
    )
    scheduler = make_lr_scheduler(optimizer, args, args.steps)
    policy.train()
    planner.train()
    heads.train()

    episodes_tasks = episode_task_descriptions(dataset)

    for step, batch in enumerate(cycle(dataloader), start=1):
        condition_batch, stale_offsets = async_condition_batch(
            dataset, batch, episodes_tasks, args.async_condition_interval
        )
        batch = move_batch(batch, device)
        condition_batch = move_batch(condition_batch, device)
        inputs = qwen_inputs(planner, condition_batch, args)
        z_subgoal = planner(**inputs).to(device=device, dtype=module_dtype(heads))
        outputs = heads(z_subgoal)
        policy_loss, _ = policy(prepare_policy_batch(batch, outputs.policy_condition))
        with torch.no_grad():
            aux_losses = heads.losses(
                outputs,
                aux_targets(condition_batch),
                waypoint_weight=weights.waypoint,
                gripper_weight=weights.gripper,
                stage_weight=weights.stage,
            )
        loss = weights.policy * policy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(iter_parameters([policy, planner, heads]), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0:
            print(
                "[stage3/A3-no-aux] "
                f"step={step} loss={loss.item():.6f} "
                f"policy={policy_loss.item():.6f} aux_monitor={aux_losses['loss'].item():.6f} "
                f"async_K={args.async_condition_interval} stale_mean={stale_offsets.float().mean().item():.2f}",
                flush=True,
            )
        if step % args.save_every == 0:
            save_policy(policy, stage_dir / f"checkpoint_{step:06d}" / "policy")
            save_system2_bundle(planner, heads, stage_dir / f"checkpoint_{step:06d}" / "system2", step)
        if step >= args.steps:
            break

    save_policy(policy, stage_dir / "final" / "policy")
    save_system2_bundle(planner, heads, stage_dir / "final" / "system2", args.steps)
    save_training_state(
        stage_dir,
        args.steps,
        {
            "joint_loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "aux_monitor_loss": float(aux_losses["loss"].detach().cpu()),
            "joint_aux_weight": 0.0,
            "async_condition_interval": int(args.async_condition_interval),
        },
    )


if __name__ == "__main__":
    main()
