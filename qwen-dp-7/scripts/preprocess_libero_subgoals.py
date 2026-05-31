#!/usr/bin/env python
"""Precompute System-2 auxiliary targets for LeRobot LIBERO datasets.

The script augments each data parquet with:
  - target_waypoint: future end-effector pose from observation.state[:7]
  - target_gripper_state: binary future gripper closed/open label
  - target_stage_class: heuristic approach/manipulation/retraction label

It writes a new LeRobot-style dataset directory and leaves the source dataset
untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WAYPOINT_KEY = "target_waypoint"
GRIPPER_KEY = "target_gripper_state"
STAGE_KEY = "target_stage_class"


@dataclass
class RunningStats:
    count: int = 0
    sum: np.ndarray | None = None
    sumsq: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        values = values.astype(np.float64, copy=False)

        if self.sum is None:
            width = values.shape[1]
            self.sum = np.zeros(width, dtype=np.float64)
            self.sumsq = np.zeros(width, dtype=np.float64)
            self.minimum = np.full(width, np.inf, dtype=np.float64)
            self.maximum = np.full(width, -np.inf, dtype=np.float64)

        self.count += values.shape[0]
        self.sum += values.sum(axis=0)
        self.sumsq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def as_lerobot_stats(self) -> dict[str, list[float] | list[int]]:
        if self.count == 0 or self.sum is None or self.sumsq is None:
            raise ValueError("Cannot serialize empty stats.")

        mean = self.sum / self.count
        variance = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": [self.count],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/lerobot_libero_10"),
        help="Source LeRobot dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/lerobot_libero_10_subgoals"),
        help="Destination augmented dataset directory.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Future horizon K in frames for waypoint and gripper targets.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.35,
        help="Absolute action threshold for binary gripper labels.",
    )
    parser.add_argument(
        "--closed-sign",
        choices=("positive", "negative"),
        default="positive",
        help=(
            "Which action sign means closed/clamping. LIBERO-10 demonstrations "
            "begin with -1 and usually switch to +1 at grasp-like events, so "
            "'positive' is the practical default for these auxiliary labels."
        ),
    )
    parser.add_argument(
        "--min-approach-frames",
        type=int,
        default=5,
        help="Ignore close events before this frame when finding approach end.",
    )
    parser.add_argument(
        "--min-stage-frames",
        type=int,
        default=5,
        help="Minimum length for each inferred behavioral stage.",
    )
    parser.add_argument(
        "--terminal-radius",
        type=float,
        default=0.025,
        help="EEF position radius used as fallback to detect terminal settling.",
    )
    parser.add_argument(
        "--terminal-window",
        type=int,
        default=8,
        help="Consecutive frames inside terminal radius for fallback retraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze episodes and print summary without writing an output dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir first if it already exists.",
    )
    return parser.parse_args()


def data_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))


def copy_dataset_shell(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    def ignore_data(_dir: str, names: list[str]) -> set[str]:
        return {"data"} if "data" in names else set()

    shutil.copytree(input_dir, output_dir, ignore=ignore_data)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)


def is_closed(gripper_action: np.ndarray, threshold: float, closed_sign: str) -> np.ndarray:
    if closed_sign == "positive":
        return gripper_action >= threshold
    return gripper_action <= -threshold


def first_false_to_true(close_signal: np.ndarray, min_frame: int) -> int | None:
    if len(close_signal) == 0:
        return None
    min_frame = min(max(min_frame, 0), len(close_signal) - 1)
    starts = np.flatnonzero((~close_signal[:-1]) & close_signal[1:]) + 1
    starts = starts[starts >= min_frame]
    if len(starts):
        return int(starts[0])
    closed = np.flatnonzero(close_signal[min_frame:]) + min_frame
    return int(closed[0]) if len(closed) else None


def last_true_to_false(close_signal: np.ndarray, after: int) -> int | None:
    if len(close_signal) < 2:
        return None
    ends = np.flatnonzero(close_signal[:-1] & (~close_signal[1:])) + 1
    ends = ends[ends > after]
    return int(ends[-1]) if len(ends) else None


def terminal_settle_start(positions: np.ndarray, after: int, radius: float, window: int) -> int | None:
    if len(positions) == 0:
        return None
    final_pos = positions[-1]
    distances = np.linalg.norm(positions - final_pos, axis=1)
    inside = distances <= radius
    start_min = max(after, 0)
    window = max(window, 1)
    for start in range(start_min, max(start_min, len(inside) - window + 1)):
        if bool(inside[start : start + window].all()):
            return start
    return None


def infer_stage_labels(
    states: np.ndarray,
    actions: np.ndarray,
    threshold: float,
    closed_sign: str,
    min_approach_frames: int,
    min_stage_frames: int,
    terminal_radius: float,
    terminal_window: int,
) -> tuple[np.ndarray, int, int]:
    n_frames = len(states)
    if n_frames == 0:
        return np.empty(0, dtype=np.int64), 0, 0

    close_signal = is_closed(actions[:, -1], threshold, closed_sign)
    approach_end = first_false_to_true(close_signal, min_approach_frames)
    if approach_end is None:
        approach_end = int(round(0.35 * n_frames))
    approach_end = int(np.clip(approach_end, min_stage_frames, max(min_stage_frames, n_frames - 2)))

    retraction_start = last_true_to_false(close_signal, approach_end + min_stage_frames)
    fallback_start = terminal_settle_start(
        states[:, :3],
        after=approach_end + min_stage_frames,
        radius=terminal_radius,
        window=terminal_window,
    )
    if retraction_start is None:
        retraction_start = fallback_start
    elif fallback_start is not None and fallback_start < retraction_start:
        retraction_start = fallback_start
    if retraction_start is None:
        retraction_start = int(round(0.80 * n_frames))

    retraction_start = int(
        np.clip(retraction_start, approach_end + min_stage_frames, max(approach_end + min_stage_frames, n_frames - 1))
    )

    stages = np.zeros(n_frames, dtype=np.int64)
    stages[approach_end:retraction_start] = 1
    stages[retraction_start:] = 2
    return stages, approach_end, retraction_start


def augment_episode(
    episode: pd.DataFrame,
    horizon: int,
    threshold: float,
    closed_sign: str,
    min_approach_frames: int,
    min_stage_frames: int,
    terminal_radius: float,
    terminal_window: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    episode = episode.sort_values("frame_index").copy()
    states = np.stack(episode["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(episode["action"].to_numpy()).astype(np.float32)
    n_frames = len(episode)
    future_indices = np.minimum(np.arange(n_frames) + horizon, n_frames - 1)

    target_waypoint = states[future_indices, :7].astype(np.float32)
    target_gripper = is_closed(actions[future_indices, -1], threshold, closed_sign).astype(np.int64)
    target_stage, approach_end, retraction_start = infer_stage_labels(
        states=states,
        actions=actions,
        threshold=threshold,
        closed_sign=closed_sign,
        min_approach_frames=min_approach_frames,
        min_stage_frames=min_stage_frames,
        terminal_radius=terminal_radius,
        terminal_window=terminal_window,
    )

    episode[WAYPOINT_KEY] = list(target_waypoint)
    episode[GRIPPER_KEY] = target_gripper
    episode[STAGE_KEY] = target_stage
    summary = {
        "episode_index": int(episode["episode_index"].iloc[0]),
        "length": n_frames,
        "approach_end": approach_end,
        "retraction_start": retraction_start,
        "closed_frames": int(target_gripper.sum()),
    }
    return episode, summary


def update_info_json(output_dir: Path, fps: int) -> None:
    info_path = output_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.setdefault("features", {})
    features[WAYPOINT_KEY] = {
        "dtype": "float32",
        "shape": [7],
        "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw"]},
        "fps": fps,
    }
    features[GRIPPER_KEY] = {
        "dtype": "int64",
        "shape": [1],
        "names": {"classes": ["open_or_moving", "closed_or_clamping"]},
        "fps": fps,
    }
    features[STAGE_KEY] = {
        "dtype": "int64",
        "shape": [1],
        "names": {"classes": ["approach", "manipulation", "retraction"]},
        "fps": fps,
    }

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def update_stats_json(output_dir: Path, stats: dict[str, RunningStats]) -> None:
    stats_path = output_dir / "meta" / "stats.json"
    with stats_path.open("r", encoding="utf-8") as f:
        stats_json = json.load(f)
    for key, running in stats.items():
        stats_json[key] = running.as_lerobot_stats()
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)
        f.write("\n")


def process(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    files = data_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_dir / 'data'}")

    info = json.load((input_dir / "meta" / "info.json").open("r", encoding="utf-8"))
    fps = int(info.get("fps", 10))
    print(f"Input: {input_dir}")
    print(f"Episodes: {info.get('total_episodes')}  frames: {info.get('total_frames')}  fps: {fps}")
    print(f"Files: {len(files)}  horizon: {args.horizon}  closed_sign: {args.closed_sign}")

    if not args.dry_run:
        copy_dataset_shell(input_dir, output_dir, args.overwrite)

    stats = {
        WAYPOINT_KEY: RunningStats(),
        GRIPPER_KEY: RunningStats(),
        STAGE_KEY: RunningStats(),
    }
    summaries: list[dict[str, int]] = []

    for file_path in files:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        augmented_parts = []
        for _, episode in df.groupby("episode_index", sort=True):
            augmented, summary = augment_episode(
                episode=episode,
                horizon=args.horizon,
                threshold=args.gripper_threshold,
                closed_sign=args.closed_sign,
                min_approach_frames=args.min_approach_frames,
                min_stage_frames=args.min_stage_frames,
                terminal_radius=args.terminal_radius,
                terminal_window=args.terminal_window,
            )
            augmented_parts.append(augmented)
            summaries.append(summary)

            stats[WAYPOINT_KEY].update(np.stack(augmented[WAYPOINT_KEY].to_numpy()))
            stats[GRIPPER_KEY].update(augmented[GRIPPER_KEY].to_numpy())
            stats[STAGE_KEY].update(augmented[STAGE_KEY].to_numpy())

        augmented_df = pd.concat(augmented_parts).sort_values("index")
        if not args.dry_run:
            rel_path = file_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(augmented_df, preserve_index=False), out_path)

    lengths = np.array([s["length"] for s in summaries])
    approach = np.array([s["approach_end"] for s in summaries])
    retract = np.array([s["retraction_start"] for s in summaries])
    print(
        "Stage summary: "
        f"approach_end mean={approach.mean():.1f}, "
        f"retraction_start mean={retract.mean():.1f}, "
        f"episode_len mean={lengths.mean():.1f}"
    )
    print("First five episode cuts:")
    for summary in summaries[:5]:
        print(
            f"  ep={summary['episode_index']:03d} len={summary['length']} "
            f"approach_end={summary['approach_end']} "
            f"retraction_start={summary['retraction_start']} "
            f"future_closed_frames={summary['closed_frames']}"
        )

    if not args.dry_run:
        update_info_json(output_dir, fps=fps)
        update_stats_json(output_dir, stats=stats)
        print(f"Wrote augmented dataset: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.horizon < 0:
        raise ValueError("--horizon must be non-negative")
    process(args)


if __name__ == "__main__":
    main()
