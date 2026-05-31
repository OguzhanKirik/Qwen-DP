"""LeRobot Diffusion Policy with Qwen-DP subgoal conditioning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    DiffusionRgbEncoder,
    _make_noise_scheduler,
)
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


QWEN_DP_POLICY_CONDITION = "qwen_dp.policy_condition"


@PreTrainedConfig.register_subclass("qwen_dp_diffusion")
@dataclass
class System1ActorConfig(DiffusionConfig):
    """Diffusion config extended with a projected Qwen-DP subgoal condition."""

    qwen_condition_dim: int = 512
    qwen_condition_key: str = QWEN_DP_POLICY_CONDITION


class System1ActorDiffusionModel(DiffusionModel):
    """DiffusionModel variant that appends Qwen-DP context to global FiLM conditioning."""

    def __init__(self, config: System1ActorConfig):
        super(DiffusionModel, self).__init__()
        self.config = config

        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = torch.nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        base_global_cond_dim = global_cond_dim * config.n_obs_steps
        total_global_cond_dim = base_global_cond_dim + config.qwen_condition_dim
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=total_global_cond_dim)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        base_global_cond = super()._prepare_global_conditioning(batch)
        qwen_condition = batch.get(self.config.qwen_condition_key)
        if qwen_condition is None:
            raise KeyError(
                f"Missing '{self.config.qwen_condition_key}' in batch. "
                "Compute it with AuxiliaryHeads(...).policy_condition before calling the policy."
            )
        if qwen_condition.ndim == 3:
            qwen_condition = qwen_condition[:, -1]
        if qwen_condition.ndim != 2:
            raise ValueError(
                f"Expected {self.config.qwen_condition_key} shape [B, D] or [B, T, D], "
                f"got {tuple(qwen_condition.shape)}."
            )
        if qwen_condition.shape[0] != base_global_cond.shape[0]:
            raise ValueError(
                f"Batch mismatch: base condition has B={base_global_cond.shape[0]}, "
                f"Qwen condition has B={qwen_condition.shape[0]}."
            )
        if qwen_condition.shape[-1] != self.config.qwen_condition_dim:
            raise ValueError(
                f"Expected Qwen condition dim {self.config.qwen_condition_dim}, "
                f"got {qwen_condition.shape[-1]}."
            )
        qwen_condition = qwen_condition.to(device=base_global_cond.device, dtype=base_global_cond.dtype)
        return torch.cat([base_global_cond, qwen_condition], dim=-1)


class System1Actor(DiffusionPolicy):
    """LeRobot DiffusionPolicy with an extra Qwen-DP condition vector.

    The policy still uses LeRobot's RGB encoder, state encoder, action diffusion
    loss, schedulers, queues, and FiLM-conditioned U-Net. The only System-1
    architectural change is appending `qwen_dp.policy_condition` to the global
    conditioning vector consumed by the existing FiLM residual blocks.
    """

    config_class = System1ActorConfig
    name = "qwen_dp_diffusion"

    def __init__(self, config: System1ActorConfig, **kwargs):
        super(DiffusionPolicy, self).__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = System1ActorDiffusionModel(config)
        self.reset()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
            self.config.qwen_condition_key: deque(maxlen=self.config.n_obs_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)
        if self.config.qwen_condition_key not in batch:
            raise KeyError(f"Missing '{self.config.qwen_condition_key}' in rollout batch.")

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.qwen_condition_key not in batch:
            raise KeyError(f"Missing '{self.config.qwen_condition_key}' in training batch.")
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        return loss, None
