#!/usr/bin/env python
"""Check Step-5 Qwen System-2 initialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_dp import System2Planner, System2PlannerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/Qwen2-VL-7B")
    parser.add_argument("--subgoal-token", default="<SUBGOAL>")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_map = None if args.device_map == "none" else args.device_map
    system2 = System2Planner(
        System2PlannerConfig(
            model_path=args.model_path,
            subgoal_token=args.subgoal_token,
            torch_dtype=args.torch_dtype,
            device_map=device_map,
        )
    )

    trainable = system2.trainable_parameters()
    total_params = sum(param.numel() for param in system2.parameters())
    trainable_params = sum(param.numel() for _, param in trainable)

    print(f"model_path: {Path(args.model_path).resolve()}")
    print(f"subgoal_token: {args.subgoal_token}")
    print(f"subgoal_token_id: {system2.subgoal_token_id}")
    print(f"hidden_size: {system2.hidden_size}")
    print(f"total_params: {total_params:,}")
    print(f"trainable_parameter_tensors: {len(trainable)}")
    print(f"trainable_params_before_gradient_mask: {trainable_params:,}")
    for name, param in trainable:
        print(f"  trainable tensor: {name} shape={tuple(param.shape)}")

    prompt = f"Describe the robot subgoal. {args.subgoal_token}"
    inputs = system2.processor(text=[prompt], return_tensors="pt")
    device = system2.subgoal_embedding().device
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    print(f"prompt_contains_subgoal: {bool((inputs['input_ids'] == system2.subgoal_token_id).any())}")
    print("step5_check: ok")


if __name__ == "__main__":
    main()
