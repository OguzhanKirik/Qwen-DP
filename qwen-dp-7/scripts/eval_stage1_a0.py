#!/usr/bin/env python
"""
Full evaluation of Model A0: System 1 (Diffusion Policy) with task-ID conditioning.

If a task_embed.pt file is found alongside the checkpoint (produced by Stage 1 v3
training), the correct learned task embedding is injected per task — matching the
conditioning the policy was actually trained with.

Falls back to zero conditioning automatically if task_embed.pt is not present
(e.g. when evaluating an older Stage 1 checkpoint trained without task embeddings).
"""

import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qwen-dp-7"))
sys.path.insert(0, str(ROOT / "lerobot" / "src"))

import argparse
import json
from lerobot.configs.eval import EvalConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import run_one, _align_libero_eval_config
from qwen_dp import System1Actor, QWEN_DP_POLICY_CONDITION


class System1ActorTaskEmbedCondition(System1Actor):
    """Wraps System1Actor to inject a task-ID embedding (or zeros) as conditioning.

    If task_embed.pt exists in the checkpoint directory, the learned embedding for
    the current task is injected. Call set_task_id(task_id) before each task's
    rollout batch. Falls back to zeros when no embedding file is found.
    """

    def __init__(self, *args, task_embed_path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_embed: torch.nn.Embedding | None = None
        self._current_task_id: int = 0

        if task_embed_path is not None and task_embed_path.exists():
            state = torch.load(task_embed_path, map_location="cpu", weights_only=True)
            num_tasks, embed_dim = next(iter(state.values())).shape
            self._task_embed = torch.nn.Embedding(num_tasks, embed_dim)
            self._task_embed.load_state_dict(state)
            print(f"Loaded task embeddings from {task_embed_path} "
                  f"({num_tasks} tasks, dim={embed_dim})")
        else:
            print("No task_embed.pt found — using zero conditioning for A0 eval.")

    def set_task_id(self, task_id: int) -> None:
        self._current_task_id = task_id

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # Normalise camera key names from LIBERO env to policy expectations
        if "observation.images.image2" in batch:
            batch["observation.images.wrist_image"] = batch.pop("observation.images.image2")
        if "observation.images.agentview_image" in batch:
            batch["observation.images.image"] = batch.pop("observation.images.agentview_image")
        if "observation.images.robot0_eye_in_hand_image" in batch:
            batch["observation.images.wrist_image"] = batch.pop(
                "observation.images.robot0_eye_in_hand_image"
            )

        # Find reference tensor for batch metadata
        reference = next(
            (v for v in batch.values() if isinstance(v, Tensor)), None
        )
        if reference is None:
            raise ValueError("Batch contains no tensors.")

        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype

        if self._task_embed is not None:
            task_ids = torch.full(
                (batch_size,), self._current_task_id, dtype=torch.long, device="cpu"
            )
            condition = self._task_embed(task_ids).to(device=device, dtype=dtype)
        else:
            condition = torch.zeros(
                batch_size, self.config.qwen_condition_dim, device=device, dtype=dtype
            )

        batch[QWEN_DP_POLICY_CONDITION] = condition
        return super().select_action(batch, noise=noise)


def eval_all_tasks_with_task_id(
    envs: dict,
    policy: System1ActorTaskEmbedCondition,
    *,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path,
    start_seed: int,
) -> dict:
    """Iterate over all tasks, inject the correct task embedding, run rollouts."""
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    group_acc: dict = defaultdict(lambda: {"sum_rewards": [], "max_rewards": [], "successes": []})
    overall = {"sum_rewards": [], "max_rewards": [], "successes": []}
    per_task_infos = []
    t0 = time.time()

    runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=False,
        start_seed=start_seed,
        debug_log_rollout=False,
        debug_logged_steps=0,
    )

    for task_group, task_id, env in tasks:
        print(f"  → task_group={task_group}  task_id={task_id}  "
              f"embed={'learned' if policy._task_embed is not None else 'zeros'}")
        policy.set_task_id(task_id)
        tg, tid, metrics = runner(task_group, task_id, env)

        for key in ("sum_rewards", "max_rewards", "successes"):
            val = metrics.get(key, [])
            lst = val if isinstance(val, list) else [val]
            group_acc[tg][key].extend(lst)
            overall[key].extend(lst)

        per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    eval_s = time.time() - t0
    n_eps = len(overall["successes"])

    def _pct(lst):
        return 100 * sum(lst) / max(len(lst), 1)

    def _avg(lst):
        return sum(lst) / max(len(lst), 1)

    return {
        "overall": {
            "pc_success": _pct(overall["successes"]),
            "avg_sum_reward": _avg(overall["sum_rewards"]),
            "avg_max_reward": _avg(overall["max_rewards"]),
            "n_episodes": n_eps,
            "eval_s": eval_s,
            "eval_ep_s": eval_s / max(n_eps, 1),
        },
        "per_group": {
            tg: {
                "pc_success": _pct(v["successes"]),
                "n_episodes": len(v["successes"]),
            }
            for tg, v in group_acc.items()
        },
        "per_task": per_task_infos,
    }


def main():
    parser_args = argparse.ArgumentParser(description="Evaluate Stage 1 A0")
    parser_args.add_argument("--checkpoint", type=str, required=True)
    parser_args.add_argument("--task-embed-path", type=str, default=None,
                             help="Path to task_embed.pt. Auto-detected from checkpoint "
                                  "dir if omitted.")
    parser_args.add_argument("--env_type", type=str, default="libero")
    parser_args.add_argument("--env_task", type=str, default="libero_10")
    parser_args.add_argument("--gripper-mode", choices=("openvla", "lerobot"),
                             default="lerobot")
    parser_args.add_argument("--n_episodes", type=int, default=50)
    parser_args.add_argument("--batch_size", type=int, default=10)
    parser_args.add_argument("--max_videos", type=int, default=10)
    parser_args.add_argument("--device", type=str, default="cuda")
    parser_args.add_argument("--output_dir", type=str,
                             default="slurm_eval/results/stage1/a0")
    parser_args.add_argument("--seed", type=int, default=1000)
    args = parser_args.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "rollout_videos"
    videos_dir.mkdir(exist_ok=True)

    # Auto-detect task_embed.pt from checkpoint directory
    checkpoint_dir = Path(args.checkpoint)
    if args.task_embed_path is not None:
        task_embed_path = Path(args.task_embed_path)
    else:
        task_embed_path = checkpoint_dir / "task_embed.pt"

    print("=" * 60)
    print("Model A0 Evaluation: System 1 with task-ID conditioning")
    print("=" * 60)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Task embed:     {task_embed_path if task_embed_path.exists() else 'not found (zeros)'}")
    print(f"Environment:    {args.env_type}/{args.env_task}")
    print(f"Episodes:       {args.n_episodes}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Gripper mode:   {args.gripper_mode}")
    print(f"Output:         {output_dir}")
    print("=" * 60)

    print("\nLoading policy...")
    policy = System1ActorTaskEmbedCondition.from_pretrained(
        args.checkpoint,
        task_embed_path=task_embed_path,
    )
    policy = policy.to(args.device)
    policy.eval()
    print(f"Policy loaded — type={policy.config.type}, "
          f"condition_dim={policy.config.qwen_condition_dim}")

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
        gym_kwargs: dict = field(
            default_factory=lambda: {"render_mode": "rgb_array",
                                     "obs_type": "pixels_agent_pos"}
        )

    cfg = type("obj", (object,), {
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
    })()

    _align_libero_eval_config(cfg)

    print(f"\nInitialising {args.env_task} environments...")
    envs = make_env(cfg.env, n_envs=args.batch_size)

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        cfg.env, policy.config
    )
    preprocessor, postprocessor = make_pre_post_processors(policy.config)
    print("Preprocessors ready.")

    print(f"\n{'=' * 60}")
    print(f"Running evaluation: {args.n_episodes} episodes per task...")
    print(f"{'=' * 60}\n")

    eval_results = eval_all_tasks_with_task_id(
        envs=envs,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=args.n_episodes,
        max_episodes_rendered=args.max_videos,
        videos_dir=videos_dir,
        start_seed=args.seed,
    )

    overall = eval_results["overall"]
    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Success Rate:  {overall['pc_success']:.2f}%")
    print(f"  Avg Reward:    {overall['avg_sum_reward']:.3f}")
    print(f"  Episodes:      {overall['n_episodes']}")
    print(f"  Eval time:     {overall['eval_s']:.1f}s "
          f"({overall['eval_ep_s']:.2f}s/episode)")

    if "per_group" in eval_results:
        print("\nPer-task-group:")
        for group, m in eval_results["per_group"].items():
            print(f"  {group}: {m['pc_success']:.2f}%  ({m['n_episodes']} eps)")

    results_file = output_dir / "eval_info.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    print(f"Videos saved to:  {videos_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
