"""Projector and auxiliary heads for Qwen-DP System-2 alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AuxiliaryHeadsConfig:
    z_subgoal_dim: int = 3584
    policy_condition_dim: int = 512
    projector_hidden_dim: int = 1024
    aux_hidden_dim: int | None = None
    dropout: float = 0.0
    waypoint_dim: int = 7
    num_stage_classes: int = 3


@dataclass
class AuxiliaryHeadOutput:
    policy_condition: Tensor
    waypoint: Tensor
    gripper_logit: Tensor
    stage_logits: Tensor


@dataclass
class AuxiliaryTargets:
    target_waypoint: Tensor
    target_gripper_state: Tensor
    target_stage_class: Tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AuxiliaryHeads(nn.Module):
    """Step-6 bridge from Qwen `<SUBGOAL>` hidden state to policy + aux targets.

    Inputs:
        z_subgoal: final-layer hidden state at the `<SUBGOAL>` token,
            shape `[batch, z_subgoal_dim]`.

    Outputs:
        policy_condition: projected vector for System-1 conditioning.
        waypoint: regression prediction for `[x, y, z, rx, ry, rz, rw]`.
        gripper_logit: raw binary logit for closed/clamping.
        stage_logits: raw 3-class logits for approach/manipulation/retraction.
    """

    def __init__(self, config: AuxiliaryHeadsConfig = AuxiliaryHeadsConfig()):
        super().__init__()
        self.config = config
        aux_hidden_dim = config.aux_hidden_dim or config.projector_hidden_dim

        self.projector = MLP(
            input_dim=config.z_subgoal_dim,
            hidden_dim=config.projector_hidden_dim,
            output_dim=config.policy_condition_dim,
            dropout=config.dropout,
        )
        self.waypoint_head = MLP(
            input_dim=config.policy_condition_dim,
            hidden_dim=aux_hidden_dim,
            output_dim=config.waypoint_dim,
            dropout=config.dropout,
        )
        self.gripper_head = nn.Linear(config.policy_condition_dim, 1)
        self.stage_head = nn.Linear(config.policy_condition_dim, config.num_stage_classes)

    def forward(self, z_subgoal: Tensor) -> AuxiliaryHeadOutput:
        if z_subgoal.ndim != 2:
            raise ValueError(f"Expected z_subgoal shape [batch, dim], got {tuple(z_subgoal.shape)}")
        if z_subgoal.shape[-1] != self.config.z_subgoal_dim:
            raise ValueError(
                f"Expected z_subgoal dim {self.config.z_subgoal_dim}, got {z_subgoal.shape[-1]}"
            )

        policy_condition = self.projector(z_subgoal)
        return AuxiliaryHeadOutput(
            policy_condition=policy_condition,
            waypoint=self.waypoint_head(policy_condition),
            gripper_logit=self.gripper_head(policy_condition).squeeze(-1),
            stage_logits=self.stage_head(policy_condition),
        )

    def losses(
        self,
        outputs: AuxiliaryHeadOutput,
        targets: AuxiliaryTargets,
        waypoint_weight: float = 1.0,
        gripper_weight: float = 1.0,
        stage_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        waypoint_target = targets.target_waypoint.to(outputs.waypoint.device, dtype=outputs.waypoint.dtype)
        gripper_target = targets.target_gripper_state.to(
            outputs.gripper_logit.device, dtype=outputs.gripper_logit.dtype
        )
        stage_target = targets.target_stage_class.to(outputs.stage_logits.device, dtype=torch.long)

        waypoint_loss = F.mse_loss(outputs.waypoint, waypoint_target)
        gripper_loss = F.binary_cross_entropy_with_logits(outputs.gripper_logit, gripper_target)
        stage_loss = F.cross_entropy(outputs.stage_logits, stage_target)
        total = (
            waypoint_weight * waypoint_loss
            + gripper_weight * gripper_loss
            + stage_weight * stage_loss
        )
        return {
            "loss": total,
            "waypoint_loss": waypoint_loss,
            "gripper_loss": gripper_loss,
            "stage_loss": stage_loss,
        }
