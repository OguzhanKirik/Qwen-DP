#!/usr/bin/env python
"""Check Step-6 projector and auxiliary heads."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_dp import AuxiliaryTargets, AuxiliaryHeadsConfig, AuxiliaryHeads  # noqa: E402


def main() -> None:
    config = AuxiliaryHeadsConfig(
        z_subgoal_dim=1536,
        policy_condition_dim=512,
        projector_hidden_dim=1024,
    )
    heads = AuxiliaryHeads(config)
    z_subgoal = torch.randn(4, config.z_subgoal_dim)
    outputs = heads(z_subgoal)
    targets = AuxiliaryTargets(
        target_waypoint=torch.randn(4, 7),
        target_gripper_state=torch.tensor([0, 1, 0, 1]),
        target_stage_class=torch.tensor([0, 1, 2, 1]),
    )
    losses = heads.losses(outputs, targets)

    print(f"policy_condition: {tuple(outputs.policy_condition.shape)}")
    print(f"waypoint: {tuple(outputs.waypoint.shape)}")
    print(f"gripper_logit: {tuple(outputs.gripper_logit.shape)}")
    print(f"stage_logits: {tuple(outputs.stage_logits.shape)}")
    print(f"loss: {float(losses['loss'].detach()):.6f}")
    print("step6_check: ok")


if __name__ == "__main__":
    main()
