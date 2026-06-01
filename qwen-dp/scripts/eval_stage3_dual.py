#!/usr/bin/env python
"""Evaluate dual-system Qwen-DP variants A2, A3, A4, and A5 on LIBERO.

Recovery & compounding diagnostics (enabled when --perturb-at-step > 0):
  --perturb-at-step   Step at which random actions are injected (0 = disabled).
  --perturb-steps     How many consecutive random-action steps to inject.
  --perturb-scale     Magnitude of random actions (fraction of action range).

These flags answer two questions:
  Q2a  Perturbation recovery — does System 2 re-plan after forced drift?
       Compare A4 vs A5 success rate and steps-to-recovery under perturbation.
  Q2b  Error compounding — does drift accumulate faster without re-conditioning?
       Compare per-step state drift across A1/A2/A4/A5 on clean rollouts
       (perturb-at-step=0). Faster-growing drift = more compounding.

Per-episode metrics are written to <output_dir>/episode_metrics.json alongside
the standard eval_info.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qwen-dp"))
sys.path.insert(0, str(ROOT / "lerobot" / "src"))
sys.path.insert(0, str(ROOT / "qwen-dp" / "scripts"))

from lerobot.configs.default import EvalConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import _align_libero_eval_config, eval_policy_all
from lerobot.utils.constants import ACTION
from qwen_dp import QWEN_DP_POLICY_CONDITION, System1Actor
from train_common import (
    load_system2_bundle,
    make_aux_heads,
    make_system2,
    module_dtype,
    qwen_inputs,
)


def normalize_libero_camera_keys(batch: dict) -> dict:
    batch = dict(batch)
    if "observation.images.image2" in batch:
        batch["observation.images.wrist_image"] = batch.pop("observation.images.image2")
    if "observation.images.agentview_image" in batch:
        batch["observation.images.image"] = batch.pop("observation.images.agentview_image")
    if "observation.images.robot0_eye_in_hand_image" in batch:
        batch["observation.images.wrist_image"] = batch.pop("observation.images.robot0_eye_in_hand_image")
    return batch


class DualSystemEvalPolicy(System1Actor):
    """System1 policy wrapper that injects System2-derived conditioning."""

    def __init__(
        self,
        policy_checkpoint: str,
        system2_checkpoint: str,
        args: argparse.Namespace,
    ):
        base_policy = System1Actor.from_pretrained(policy_checkpoint)
        super().__init__(base_policy.config)
        self.load_state_dict(base_policy.state_dict())
        self.to(args.device)

        self.variant = args.variant
        self.update_interval = max(args.update_interval, 1)
        self.recovery_threshold = args.recovery_threshold
        self.args = args

        self.planner = make_system2(args).to(args.device)
        self.heads = make_aux_heads(args, self.planner).to(args.device)
        load_system2_bundle(self.planner, self.heads, Path(system2_checkpoint))
        self.planner.eval()
        self.heads.eval()

        self._step = 0
        self._cached_condition: Tensor | None = None
        self._last_state: Tensor | None = None
        self._num_condition_updates = 0

        # perturbation config
        self._perturb_at_step: int = getattr(args, "perturb_at_step", 0)
        self._perturb_steps: int = getattr(args, "perturb_steps", 5)
        self._perturb_scale: float = getattr(args, "perturb_scale", 1.0)
        self._action_noise_scale: float = getattr(args, "action_noise_scale", 0.0)

        # per-episode diagnostic accumulators (flushed on reset)
        self._drift_log: list[tuple[int, float]] = []       # (step, drift_from_prev)
        self._condition_update_steps: list[int] = []        # steps when S2 re-conditioned
        self._recovery_triggered: bool = False              # A5 drift-recovery fired post-perturb
        self._post_perturb_updates: int = 0                 # S2 updates after perturbation ends
        self._pre_perturb_state: Tensor | None = None       # state snapshot just before perturbation
        # Q3 bottleneck: S2 predictions and sampled state trajectory
        self._s2_predictions: list[dict] = []               # {step, waypoint, stage, gripper, state}
        self._state_trajectory: list[tuple[int, list]] = [] # (step, state) sampled every 5 steps
        self.episode_metrics: list[dict] = []               # accumulated across all episodes

    def reset(self):
        # flush previous episode before resetting state
        if getattr(self, "_step", 0) > 0:
            episode_metrics = getattr(self, "episode_metrics", None)
            if episode_metrics is None:
                episode_metrics = []
                self.episode_metrics = episode_metrics
            episode_metrics.append({
                "steps": self._step,
                "num_condition_updates": self._num_condition_updates,
                "condition_update_steps": list(self._condition_update_steps),
                "drift_log": list(self._drift_log),
                "post_perturb_s2_updates": self._post_perturb_updates,
                "recovery_triggered": self._recovery_triggered,
                "s2_predictions": list(self._s2_predictions),
                "state_trajectory": list(self._state_trajectory),
            })
        super().reset()
        self._step = 0
        self._cached_condition = None
        self._last_state = None
        self._num_condition_updates = 0
        self._drift_log = []
        self._condition_update_steps = []
        self._recovery_triggered = False
        self._post_perturb_updates = 0
        self._pre_perturb_state = None
        self._s2_predictions = []
        self._state_trajectory = []

    @torch.no_grad()
    def _compute_condition(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = qwen_inputs(self.planner, batch, self.args)
        z_subgoal = self.planner(**inputs).to(device=self.args.device, dtype=module_dtype(self.heads))
        outputs = self.heads(z_subgoal)
        self._num_condition_updates += 1
        self._condition_update_steps.append(self._step)
        perturb_end = self._perturb_at_step + self._perturb_steps
        if self._perturb_at_step > 0 and self._step >= perturb_end:
            self._post_perturb_updates += 1
            self._recovery_triggered = True
        # Q3: log S2 reasoning outputs for bottleneck analysis
        state_now = self._last_state[0].cpu().tolist() if self._last_state is not None else None
        self._s2_predictions.append({
            "step": self._step,
            "waypoint": outputs.waypoint[0].cpu().tolist(),
            "stage": int(outputs.stage_logits[0].argmax()),
            "gripper": bool((outputs.gripper_logit[0] > 0).item()),
            "state_at_prediction": state_now,
        })
        return outputs.policy_condition.to(device=self.args.device)

    def _should_update_condition(self, batch: dict[str, Tensor]) -> bool:
        if self._cached_condition is None:
            return True
        # suppress re-conditioning during perturbation window — actions are overridden
        # anyway, so any S2 output would be wasted and pollute the recovery measurement.
        # use self._step + 1 (post-increment) to match the injection check in select_action.
        perturb_end = self._perturb_at_step + self._perturb_steps
        if self._perturb_at_step > 0 and self._perturb_at_step <= self._step + 1 < perturb_end:
            return False
        if self.variant == "a2":
            return False
        if self._step % self.update_interval == 0:
            return True
        if self.variant == "a5" and self._last_state is not None and "observation.state" in batch:
            state = batch["observation.state"].detach()
            drift = torch.linalg.vector_norm(state - self._last_state.to(state.device), dim=-1)
            if bool((drift > self.recovery_threshold).any()):
                return True
        return False

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = normalize_libero_camera_keys(batch)

        perturb_end = self._perturb_at_step + self._perturb_steps

        # On the first normal step after perturbation (pre-increment step == perturb_end - 1),
        # restore _last_state to the pre-perturbation snapshot BEFORE the drift check runs.
        # This makes _should_update_condition see the full displacement in one shot,
        # so A5 re-conditions immediately at this step rather than one step later.
        if (self._perturb_at_step > 0
                and self._pre_perturb_state is not None
                and self._step == perturb_end - 1):
            self._last_state = self._pre_perturb_state

        # track per-step state drift for compounding analysis
        if "observation.state" in batch:
            state = batch["observation.state"].detach()
            if self._last_state is not None:
                drift = float(
                    torch.linalg.vector_norm(state - self._last_state.to(state.device), dim=-1).mean()
                )
                self._drift_log.append((self._step, drift))

        condition_updated = self._should_update_condition(batch)
        if condition_updated:
            self._cached_condition = self._compute_condition(batch)
            # A System2 refresh starts a new low-frequency control window. Drop
            # any remaining high-frequency System1 actions from the old condition
            # so --update-interval 4 really means a 1:4 System2:System1 ratio.
            if self._queues is not None and ACTION in self._queues:
                self._queues[ACTION].clear()

        reference = next(value for value in batch.values() if isinstance(value, Tensor))
        condition = self._cached_condition
        if condition is None:
            raise RuntimeError("System2 condition cache was not initialized.")
        if condition.shape[0] != reference.shape[0]:
            condition = condition[:1].expand(reference.shape[0], -1)

        batch[QWEN_DP_POLICY_CONDITION] = condition.to(device=reference.device, dtype=reference.dtype)
        if "observation.state" in batch:
            self._last_state = batch["observation.state"].detach().clone()
            # Q3: sample state every 5 steps for waypoint hit analysis
            if self._step % 5 == 0:
                self._state_trajectory.append((self._step, self._last_state[0].cpu().tolist()))
        self._step += 1

        action = super().select_action(batch, noise=noise)

        if self._perturb_at_step > 0:
            # snapshot state on the call just before first random action (post-increment == perturb_at)
            if self._step == self._perturb_at_step and "observation.state" in batch:
                self._pre_perturb_state = batch["observation.state"].detach().clone()

            # inject random actions during perturbation window
            if self._perturb_at_step <= self._step < perturb_end:
                random_action = torch.rand_like(action) * 2 - 1  # uniform in [-1, 1]
                action = random_action * self._perturb_scale

        if self._action_noise_scale > 0.0:
            action = action + torch.randn_like(action) * self._action_noise_scale

        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("a2", "a3", "a4", "a5"), required=True)
    parser.add_argument("--policy-checkpoint", type=str, default="outputs/stage3/final/policy")
    parser.add_argument("--system2-checkpoint", type=str, default="outputs/stage3/final/system2")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "Qwen2-VL-2B")
    parser.add_argument("--system2-device-map", default="auto")
    parser.add_argument("--system2-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--system2-lora-r", type=int, default=0)
    parser.add_argument("--system2-lora-alpha", type=float, default=16.0)
    parser.add_argument("--system2-lora-dropout", type=float, default=0.05)
    parser.add_argument("--system2-lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--policy-condition-dim", type=int, default=512)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument(
        "--prompt-template",
        default="<|vision_start|><|image_pad|><|vision_end|>Robot task: {task}. Predict the next physical subgoal. <SUBGOAL>",
    )
    parser.add_argument(
        "--qwen-image-key",
        default="observation.images.image",
        choices=("observation.images.image", "observation.images.wrist_image"),
    )
    parser.add_argument("--update-interval", type=int, default=8)
    parser.add_argument("--recovery-threshold", type=float, default=0.15)
    parser.add_argument("--perturb-at-step", type=int, default=0,
                        help="Step at which random actions are injected. 0 = disabled.")
    parser.add_argument("--perturb-steps", type=int, default=5,
                        help="Number of consecutive random-action steps to inject.")
    parser.add_argument("--perturb-scale", type=float, default=3.0,
                        help="Magnitude of random actions. Default 3.0 ensures displacement >> recovery-threshold=0.15.")
    parser.add_argument("--action-noise-scale", type=float, default=0.0,
                        help="Std of Gaussian noise added to every action. 0 = disabled. Used for Q2b compounding analysis.")
    parser.add_argument("--env_type", type=str, default="libero")
    parser.add_argument("--env_task", type=str, default="libero_10")
    parser.add_argument(
        "--gripper-mode",
        choices=("openvla", "openvla_sticky", "lerobot"),
        default="openvla",
        help=(
            "LIBERO gripper convention adapter. The downloaded lerobot/libero_10 "
            "actions use -1 for open/no-clamp and +1 for close/clamp, which matches "
            "the wrapper's openvla conversion path."
        ),
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_videos", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="slurm_eval/results/stage3/a4")
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "rollout_videos"
    videos_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Model {args.variant.upper()} Evaluation")
    print("=" * 60)
    print(f"Policy checkpoint: {args.policy_checkpoint}")
    print(f"System2 checkpoint: {args.system2_checkpoint}")
    print(f"Update interval: {args.update_interval}")
    print(f"Recovery threshold: {args.recovery_threshold}")
    print(f"Perturbation at step: {args.perturb_at_step} (0 = disabled)")
    if args.perturb_at_step > 0:
        print(f"Perturbation steps: {args.perturb_steps}, scale: {args.perturb_scale}")
    print(f"Gripper mode: {args.gripper_mode}")
    print(f"Output: {output_dir}")

    policy = DualSystemEvalPolicy(args.policy_checkpoint, args.system2_checkpoint, args)
    policy.eval()

    from dataclasses import dataclass, field

    @dataclass
    class MockEnvConfig:
        type: str = args.env_type
        task: str = args.env_task
        camera_name: str = "agentview_image,robot0_eye_in_hand_image"
        init_states: bool = True
        control_mode: str = "relative"
        gripper_mode: str = args.gripper_mode
        episode_length: int | None = None
        gym_kwargs: dict = field(default_factory=lambda: {"render_mode": "rgb_array", "obs_type": "pixels_agent_pos"})

    cfg = type(
        "obj",
        (object,),
        {
            "env": MockEnvConfig(),
            "eval": EvalConfig(n_episodes=args.n_episodes, batch_size=args.batch_size),
            "policy": policy.config,
            "output_dir": output_dir,
            "seed": args.seed,
            "rename_map": {
                "observation.images.agentview_image": "observation.images.image",
                "observation.images.robot0_eye_in_hand_image": "observation.images.wrist_image",
            },
            "use_checkpoint_stats": False,
        },
    )()
    _align_libero_eval_config(cfg)

    envs = make_env(cfg.env, n_envs=args.batch_size)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(cfg.env, policy.config)
    preprocessor, postprocessor = make_pre_post_processors(policy.config)

    eval_results = eval_policy_all(
        envs=envs,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=args.n_episodes,
        max_episodes_rendered=args.max_videos,
        videos_dir=videos_dir,
        return_episode_data=False,
        start_seed=args.seed,
        max_parallel_tasks=1,
        debug_log_rollout=False,
        debug_logged_steps=0,
    )

    eval_results["variant"] = args.variant.upper()
    eval_results["condition_update_interval"] = args.update_interval
    eval_results["recovery_threshold"] = args.recovery_threshold
    eval_results["perturb_at_step"] = args.perturb_at_step
    eval_results["perturb_steps"] = args.perturb_steps
    eval_results["perturb_scale"] = args.perturb_scale
    eval_results["action_noise_scale"] = args.action_noise_scale

    results_file = output_dir / "eval_info.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    # flush final episode and save per-episode diagnostics
    policy.reset()
    metrics_file = output_dir / "episode_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(policy.episode_metrics, f, indent=2)

    overall = eval_results["overall"]
    print("=" * 60)
    print("Evaluation complete")
    print(f"Success rate: {overall['pc_success']:.2f}%")
    print(f"Avg sum reward: {overall['avg_sum_reward']:.3f}")
    print(f"Results saved to: {results_file}")
    print(f"Episode metrics saved to: {metrics_file}")
    print(f"Videos saved to: {videos_dir}")


if __name__ == "__main__":
    main()
