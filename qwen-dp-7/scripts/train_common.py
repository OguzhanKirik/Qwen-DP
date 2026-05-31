"""Shared utilities for Phase 4 Qwen-DP training stages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, default_collate


ROOT = Path(__file__).resolve().parents[2]
QWEN_DP_ROOT = ROOT / "qwen-dp-7"
LEROBOT_SRC = ROOT / "lerobot" / "src"
sys.path.insert(0, str(QWEN_DP_ROOT))
sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from qwen_dp import (  # noqa: E402
    QWEN_DP_POLICY_CONDITION,
    AuxiliaryHeads,
    AuxiliaryHeadsConfig,
    AuxiliaryTargets,
    System1Actor,
    System1ActorConfig,
    System2Planner,
    System2PlannerConfig,
)


WAYPOINT_KEY = "target_waypoint"
GRIPPER_KEY = "target_gripper_state"
STAGE_KEY = "target_stage_class"
IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")


@dataclass(frozen=True)
class StageLossWeights:
    waypoint: float = 1.0
    gripper: float = 1.0
    stage: float = 1.0
    policy: float = 1.0
    aux: float = 1.0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "lerobot_libero_10_subgoals")
    parser.add_argument("--repo-id", default="lerobot/libero_10")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_qwen7")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=84)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--policy-condition-dim", type=int, default=512)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec", "video_reader"))
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)




def add_async_condition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--async-condition-interval",
        type=int,
        default=1,
        help=(
            "Number of 10 Hz System1/env steps covered by one System2 condition during Stage3 training. "
            "Use 4 for a 1:4 System2:System1 ratio, i.e. 2.5 Hz System2 with 10 Hz System1. "
            "The default 1 preserves the original fresh-condition training."
        ),
    )


def add_system2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "Qwen2-VL-7B")
    parser.add_argument("--system2-device-map", default="auto")
    parser.add_argument("--system2-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--system2-lora-r", type=int, default=0, help="LoRA rank for Qwen attention projections. 0 disables LoRA.")
    parser.add_argument("--system2-lora-alpha", type=float, default=16.0)
    parser.add_argument("--system2-lora-dropout", type=float, default=0.05)
    parser.add_argument("--system2-lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lr-system2", type=float, default=1e-5)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--waypoint-weight", type=float, default=1.0)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--stage-weight", type=float, default=1.0)
    parser.add_argument(
        "--prompt-template",
        default="<|vision_start|><|image_pad|><|vision_end|>Robot task: {task}. Predict the next physical subgoal. <SUBGOAL>",
    )
    parser.add_argument(
        "--qwen-image-key",
        default="observation.images.image",
        choices=IMAGE_KEYS,
        help="Camera frame sent to Qwen.",
    )


def delta_timestamps(n_obs_steps: int, horizon: int, fps: int) -> dict[str, list[float]]:
    obs = [i / fps for i in range(1 - n_obs_steps, 1)]
    actions = [i / fps for i in range(horizon)]
    return {
        OBS_STATE: obs,
        ACTION: actions,
        IMAGE_KEYS[0]: obs,
        IMAGE_KEYS[1]: obs,
        WAYPOINT_KEY: [0.0],
        GRIPPER_KEY: [0.0],
        STAGE_KEY: [0.0],
    }


def make_dataset(args: argparse.Namespace) -> LeRobotDataset:
    with open(args.dataset_root / "meta" / "info.json") as f:
        fps = int(json.load(f)["fps"])
    return LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps(args.n_obs_steps, args.horizon, fps),
        download_videos=False,
        video_backend=args.video_backend,
    )


def make_dataloader(args: argparse.Namespace, dataset: LeRobotDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


def feature_from_meta(feature: dict) -> PolicyFeature:
    dtype = feature["dtype"]
    shape = tuple(feature["shape"])
    if dtype == "video":
        shape = (shape[2], shape[0], shape[1])
        return PolicyFeature(type=FeatureType.VISUAL, shape=shape)
    if dtype == "float32":
        return PolicyFeature(type=FeatureType.STATE, shape=shape)
    raise ValueError(f"Unsupported feature for policy construction: {feature}")


def make_policy_config(args: argparse.Namespace, dataset: LeRobotDataset) -> System1ActorConfig:
    features = dataset.meta.features
    return System1ActorConfig(
        input_features={
            OBS_STATE: feature_from_meta(features[OBS_STATE]),
            IMAGE_KEYS[0]: feature_from_meta(features[IMAGE_KEYS[0]]),
            IMAGE_KEYS[1]: feature_from_meta(features[IMAGE_KEYS[1]]),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=tuple(features[ACTION]["shape"])),
        },
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        crop_shape=(args.crop_size, args.crop_size),
        down_dims=tuple(args.down_dims),
        qwen_condition_dim=args.policy_condition_dim,
        device=args.device,
    )


def episode_task_descriptions(dataset: LeRobotDataset) -> list[str]:
    task_descriptions = dataset.meta.tasks.index.tolist()
    episodes_tasks: list[str] = []
    for ep_idx in range(dataset.meta.total_episodes):
        first_frame_idx = int(dataset.meta.episodes[ep_idx]["dataset_from_index"])
        task_idx = int(dataset.hf_dataset[first_frame_idx]["task_index"])
        episodes_tasks.append(task_descriptions[task_idx])
    return episodes_tasks


def add_task_descriptions(batch: dict, episodes_tasks: list[str]) -> dict:
    batch = dict(batch)
    ep_indices = batch["episode_index"].tolist()
    batch["task"] = [episodes_tasks[int(i)] for i in ep_indices]
    return batch


def async_condition_batch(
    dataset: LeRobotDataset,
    batch: dict,
    episodes_tasks: list[str],
    interval: int,
) -> tuple[dict, Tensor]:
    """Return the low-frequency System2 anchor batch for a high-frequency policy batch.

    For interval=4, frames with episode-local frame_index 0..3 all use the
    condition computed from frame 0; 4..7 use frame 4; and so on. The current
    batch is still used for System1 observations and action targets.
    """
    interval = max(int(interval), 1)
    frame_index = batch["frame_index"].to(dtype=torch.long)
    stale_offsets = frame_index.remainder(interval)
    if interval == 1 or bool((stale_offsets == 0).all()):
        return add_task_descriptions(batch, episodes_tasks), stale_offsets

    dataset_index = batch["index"].to(dtype=torch.long)
    anchor_indices = (dataset_index - stale_offsets).tolist()
    samples = [dataset[int(index)] for index in anchor_indices]
    condition_batch = default_collate(samples)
    return add_task_descriptions(condition_batch, episodes_tasks), stale_offsets


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def prepare_policy_batch(batch: dict, condition: Tensor) -> dict:
    policy_batch = dict(batch)
    policy_batch[QWEN_DP_POLICY_CONDITION] = condition
    return policy_batch


def zero_condition(batch: dict, dim: int, device: torch.device) -> Tensor:
    return torch.zeros(batch[OBS_STATE].shape[0], dim, device=device, dtype=batch[OBS_STATE].dtype)


def aux_targets(batch: dict) -> AuxiliaryTargets:
    return AuxiliaryTargets(
        target_waypoint=batch[WAYPOINT_KEY].squeeze(1) if batch[WAYPOINT_KEY].ndim == 3 else batch[WAYPOINT_KEY],
        target_gripper_state=batch[GRIPPER_KEY].flatten(),
        target_stage_class=batch[STAGE_KEY].flatten(),
    )


def tensor_to_pil(frame: Tensor) -> Image.Image:
    frame = frame.detach().float().cpu()
    if frame.ndim == 4:
        frame = frame[-1]
    if frame.shape[0] in {1, 3}:
        frame = frame.permute(1, 2, 0)
    array = (frame.clamp(0, 1).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def qwen_inputs(planner: System2Planner, batch: dict, args: argparse.Namespace) -> dict[str, Tensor]:
    if "task" not in batch:
        raise KeyError(
            "Batch missing 'task' strings. Ensure the training loop fetches task descriptions "
            "from dataset metadata (dataset.meta.episodes) using the 'episode_index' in the batch."
        )
    texts = [args.prompt_template.format(task=task) for task in batch["task"]]
    images = [tensor_to_pil(frame) for frame in batch[args.qwen_image_key]]
    inputs = planner.processor(text=texts, images=images, padding=True, return_tensors="pt")
    device = planner.subgoal_embedding().device
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in inputs.items()}


def iter_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for module in modules:
        params.extend([param for param in module.parameters() if param.requires_grad])
    return params


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def jsonable_args(args: argparse.Namespace) -> dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            payload[key] = value
    return payload


def save_policy(policy: System1Actor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)


def make_system2(args: argparse.Namespace) -> System2Planner:
    device_map = None if args.system2_device_map == "none" else args.system2_device_map
    return System2Planner(
        System2PlannerConfig(
            model_path=str(args.model_path),
            torch_dtype=args.system2_dtype,
            device_map=device_map,
            lora_r=args.system2_lora_r,
            lora_alpha=args.system2_lora_alpha,
            lora_dropout=args.system2_lora_dropout,
            lora_target_modules=tuple(args.system2_lora_target_modules),
        )
    )


def make_aux_heads(args: argparse.Namespace, planner: System2Planner) -> AuxiliaryHeads:
    return AuxiliaryHeads(
        AuxiliaryHeadsConfig(
            z_subgoal_dim=planner.hidden_size,
            policy_condition_dim=args.policy_condition_dim,
            projector_hidden_dim=args.projector_hidden_dim,
        )
    ).to(torch.device(args.device))


def save_system2_bundle(planner: System2Planner, heads: AuxiliaryHeads, output_dir: Path, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "system2_config": asdict(planner.config),
            "aux_heads_config": asdict(heads.config),
            "subgoal_token_id": planner.subgoal_token_id,
            "subgoal_embedding": planner.subgoal_embedding().detach().cpu(),
            "lora_state_dict": planner.lora_state_dict(),
            "aux_heads_state_dict": heads.state_dict(),
        },
        output_dir / "system2_aux.pt",
    )


def load_system2_bundle(planner: System2Planner, heads: AuxiliaryHeads, checkpoint_dir: Path) -> None:
    state = torch.load(checkpoint_dir / "system2_aux.pt", map_location="cpu")
    heads.load_state_dict(state["aux_heads_state_dict"])
    planner.load_lora_state_dict(state.get("lora_state_dict", {}))
    with torch.no_grad():
        embedding = planner.model.get_input_embeddings().weight
        embedding[planner.subgoal_token_id].copy_(state["subgoal_embedding"].to(embedding.device, embedding.dtype))


def save_training_state(output_dir: Path, step: int, metrics: dict[str, float]) -> None:
    write_json(output_dir / "training_state.json", {"step": step, "metrics": metrics, "saved_at": time.time()})


def make_lr_scheduler(optimizer: Optimizer, args: argparse.Namespace, total_steps: int) -> LambdaLR | None:
    if args.lr_scheduler == "none":
        return None

    warmup_steps = max(args.warmup_steps, 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    min_lr_ratio = args.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch
