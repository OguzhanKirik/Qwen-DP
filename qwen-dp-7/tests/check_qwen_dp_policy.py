#!/usr/bin/env python
"""Check Step-7 Qwen-DP Diffusion Policy wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


QWEN_DP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = QWEN_DP_ROOT.parent
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
sys.path.insert(0, str(QWEN_DP_ROOT))
sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from qwen_dp import QWEN_DP_POLICY_CONDITION, System1ActorConfig, System1Actor  # noqa: E402


def main() -> None:
    config = System1ActorConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        crop_shape=(84, 84),
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32,
        qwen_condition_dim=512,
        device="cpu",
    )
    policy = System1Actor(config)
    policy.train()

    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 8),
        "observation.images.image": torch.rand(batch_size, config.n_obs_steps, 3, 96, 96),
        "observation.images.wrist_image": torch.rand(batch_size, config.n_obs_steps, 3, 96, 96),
        ACTION: torch.randn(batch_size, config.horizon, 7).clamp(-1, 1),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
        QWEN_DP_POLICY_CONDITION: torch.randn(batch_size, config.qwen_condition_dim),
    }
    loss, _ = policy(batch)
    print(f"loss: {float(loss.detach()):.6f}")

    global_cond = policy.diffusion._prepare_global_conditioning(
        {
            "observation.state": batch[OBS_STATE],
            "observation.images": torch.stack(
                [batch["observation.images.image"], batch["observation.images.wrist_image"]],
                dim=-4,
            ),
            QWEN_DP_POLICY_CONDITION: batch[QWEN_DP_POLICY_CONDITION],
        }
    )
    print(f"global_cond: {tuple(global_cond.shape)}")
    print(f"expected_global_cond_dim: {config.n_obs_steps * (8 + 64 * 2) + 512}")
    print("step7_check: ok")


if __name__ == "__main__":
    main()
