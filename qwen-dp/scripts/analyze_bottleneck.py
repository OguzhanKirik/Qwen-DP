#!/usr/bin/env python
"""Offline bottleneck analysis for Q3: S2 reasoning failure vs S1 execution failure.

Reads episode_metrics.json files produced by eval_stage3_dual.py and classifies
each failed episode into one of two failure modes:

  S2_REASONING  — S2 predicted a waypoint but the robot actually reached near it
                  and still failed. S2 gave plausible-but-wrong instructions.

  S1_EXECUTION  — S2 predicted a waypoint but the robot never got close to it.
                  Instructions may have been correct; S1 couldn't follow them.

Also computes:
  - Waypoint hit rate per variant (fraction of S2 predictions robot reached)
  - Stage sequence pattern at failure (was S2 stuck in wrong stage?)
  - S2 update count distribution (many updates + failure → S1 bottleneck)

Usage:
    python analyze_bottleneck.py \\
        --results a1:slurm_eval/results/stage1/a1/episode_metrics.json \\
                  a2:slurm_eval/results/stage3/a2/episode_metrics.json \\
                  a4:slurm_eval/results/stage3_async_k4/a4/episode_metrics.json \\
                  a5:slurm_eval/results/stage3_async_k4/a5/episode_metrics.json \\
        --oracle-trajectory slurm_eval/results/stage1/a0/episode_metrics.json \\
        --waypoint-hit-threshold 0.1 \\
        --lookahead 10 \\
        --output slurm_eval/results/bottleneck_analysis.json

Optional --oracle-trajectory: path to A0 episode_metrics.json. When provided,
S2 waypoint predictions are compared against the oracle (A0) state trajectory
at the same step rather than the variant's own trajectory. This separates
S2 hallucination (predicting a position A0 never visited) from S1 execution
failure (S2 was correct but robot couldn't follow). The per-episode
hallucination_score is the mean L2 distance between each S2 prediction and
the oracle position at that step.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


STAGE_NAMES = {0: "approach", 1: "manipulation", 2: "retraction"}


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def find_state_at_step(state_trajectory: list, target_step: int) -> list | None:
    """Return the state closest to target_step from the sampled trajectory."""
    if not state_trajectory:
        return None
    best = min(state_trajectory, key=lambda s: abs(s[0] - target_step))
    return best[1] if abs(best[0] - target_step) <= 10 else None


def load_oracle_trajectories(oracle_path: Path) -> list[list]:
    """Load per-episode state trajectories from an A0 episode_metrics.json."""
    with open(oracle_path) as f:
        episodes = json.load(f)
    return [ep.get("state_trajectory", []) for ep in episodes]


def analyze_episode(ep: dict, hit_threshold: float, lookahead: int,
                    oracle_trajectory: list | None = None) -> dict:
    """Compute per-episode bottleneck metrics."""
    predictions = ep.get("s2_predictions", [])
    trajectory = ep.get("state_trajectory", [])
    n_steps = ep.get("steps", 0)
    n_updates = ep.get("num_condition_updates", 0)

    waypoint_hits = []
    position_errors = []

    for pred in predictions:
        pred_step = pred["step"]
        check_step = pred_step + lookahead
        state_after = find_state_at_step(trajectory, check_step)
        if state_after is None or pred["state_at_prediction"] is None:
            continue
        # compare predicted waypoint position (first 3 dims: x,y,z)
        # against actual robot state position (first 3 dims assumed to be EE pos)
        pred_pos = pred["waypoint"][:3]
        actual_pos = state_after[:3]
        err = l2(pred_pos, actual_pos)
        position_errors.append(err)
        waypoint_hits.append(err < hit_threshold)

    hit_rate = sum(waypoint_hits) / len(waypoint_hits) if waypoint_hits else None
    mean_pos_error = sum(position_errors) / len(position_errors) if position_errors else None

    # hallucination score: compare S2 predictions against oracle (A0) trajectory.
    # high score = S2 pointed to positions oracle never visited = hallucination.
    hallucination_score: float | None = None
    if oracle_trajectory:
        oracle_errors = []
        for pred in predictions:
            oracle_state = find_state_at_step(oracle_trajectory, pred["step"] + lookahead)
            if oracle_state is None:
                continue
            oracle_errors.append(l2(pred["waypoint"][:3], oracle_state[:3]))
        hallucination_score = sum(oracle_errors) / len(oracle_errors) if oracle_errors else None

    # stage sequence at end of episode (last 3 predictions)
    final_stages = [STAGE_NAMES.get(p["stage"], "?") for p in predictions[-3:]]

    # drift accumulation: sum of drift in last 20% of episode
    drift_log = ep.get("drift_log", [])
    late_start = int(n_steps * 0.8)
    late_drift = [d for s, d in drift_log if s >= late_start]
    terminal_drift = sum(late_drift) / len(late_drift) if late_drift else 0.0

    return {
        "n_steps": n_steps,
        "n_s2_updates": n_updates,
        "waypoint_hit_rate": hit_rate,
        "mean_waypoint_position_error": mean_pos_error,
        "hallucination_score": hallucination_score,
        "terminal_drift": terminal_drift,
        "final_stages": final_stages,
        "post_perturb_s2_updates": ep.get("post_perturb_s2_updates", 0),
        "recovery_triggered": ep.get("recovery_triggered", False),
    }


def classify_failure(ep_metrics: dict, hit_threshold_rate: float = 0.5) -> str:
    """Classify a failed episode as S1_EXECUTION or S2_REASONING failure."""
    hit_rate = ep_metrics.get("waypoint_hit_rate")
    terminal_drift = ep_metrics.get("terminal_drift", 0.0)
    n_updates = ep_metrics.get("n_s2_updates", 0)
    n_steps = ep_metrics.get("n_steps", 1)

    if hit_rate is None:
        # no waypoint data — fall back to heuristics
        if terminal_drift > 0.05 or (n_updates / max(n_steps, 1)) > 0.3:
            return "S1_EXECUTION"
        return "S2_REASONING"

    # robot reached predicted waypoints but still failed → S2 gave wrong instructions
    if hit_rate >= hit_threshold_rate:
        return "S2_REASONING"
    # robot couldn't reach waypoints → S1 failed to execute
    return "S1_EXECUTION"


def analyze_variant(
    variant: str,
    metrics_path: Path,
    hit_threshold: float,
    lookahead: int,
    oracle_trajectories: list[list] | None = None,
) -> dict:
    with open(metrics_path) as f:
        episodes = json.load(f)

    per_episode = [
        analyze_episode(
            ep, hit_threshold, lookahead,
            oracle_trajectory=oracle_trajectories[i] if oracle_trajectories and i < len(oracle_trajectories) else None,
        )
        for i, ep in enumerate(episodes)
    ]

    n_total = len(per_episode)
    hit_rates = [e["waypoint_hit_rate"] for e in per_episode if e["waypoint_hit_rate"] is not None]
    pos_errors = [e["mean_waypoint_position_error"] for e in per_episode if e["mean_waypoint_position_error"] is not None]
    hallucination_scores = [e["hallucination_score"] for e in per_episode if e["hallucination_score"] is not None]
    terminal_drifts = [e["terminal_drift"] for e in per_episode]

    # failure classification (treat all episodes — success still informative for S2 quality)
    failure_classes = [classify_failure(e) for e in per_episode]
    s1_count = failure_classes.count("S1_EXECUTION")
    s2_count = failure_classes.count("S2_REASONING")

    # drift over time: bucket into 10 equal time bins, compute mean drift per bin
    drift_bins: dict[int, list[float]] = {i: [] for i in range(10)}
    for ep in episodes:
        n = ep.get("steps", 1)
        for step, drift in ep.get("drift_log", []):
            bucket = min(int(step / n * 10), 9)
            drift_bins[bucket].append(drift)
    drift_profile = {
        f"bin_{i}": sum(v) / len(v) if v else 0.0
        for i, v in drift_bins.items()
    }

    return {
        "variant": variant,
        "n_episodes": n_total,
        "mean_waypoint_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else None,
        "mean_waypoint_position_error": sum(pos_errors) / len(pos_errors) if pos_errors else None,
        "mean_hallucination_score": sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else None,
        "mean_terminal_drift": sum(terminal_drifts) / len(terminal_drifts) if terminal_drifts else 0.0,
        "failure_classification": {
            "S1_EXECUTION": s1_count,
            "S2_REASONING": s2_count,
            "S1_fraction": s1_count / n_total if n_total else 0.0,
            "S2_fraction": s2_count / n_total if n_total else 0.0,
        },
        "drift_profile_over_time": drift_profile,
        "per_episode": per_episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        metavar="VARIANT:PATH",
        required=True,
        help="Pairs of variant_name:path_to_episode_metrics.json",
    )
    parser.add_argument("--waypoint-hit-threshold", type=float, default=0.1,
                        help="L2 distance (m) below which robot is considered to have hit the waypoint.")
    parser.add_argument("--lookahead", type=int, default=10,
                        help="Steps after S2 prediction to check if robot reached waypoint.")
    parser.add_argument("--oracle-trajectory", type=str, default=None,
                        help="Path to A0 episode_metrics.json. When provided, S2 predictions are "
                             "compared against the oracle trajectory to compute a hallucination score.")
    parser.add_argument("--output", type=str, default="slurm_eval/results/bottleneck_analysis.json")
    args = parser.parse_args()

    oracle_trajectories = None
    if args.oracle_trajectory:
        oracle_path = Path(args.oracle_trajectory)
        if oracle_path.exists():
            print(f"Loading oracle trajectories from {oracle_path} ...")
            oracle_trajectories = load_oracle_trajectories(oracle_path)
        else:
            print(f"  [WARN] --oracle-trajectory path not found: {oracle_path}, skipping hallucination scoring.")

    all_results = {}
    for entry in args.results:
        variant, path_str = entry.split(":", 1)
        path = Path(path_str)
        if not path.exists():
            print(f"  [SKIP] {variant}: {path} not found")
            continue
        print(f"Analyzing {variant} from {path} ...")
        all_results[variant] = analyze_variant(
            variant, path, args.waypoint_hit_threshold, args.lookahead,
            oracle_trajectories=oracle_trajectories,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # print summary table
    has_hallucination = any(r["mean_hallucination_score"] is not None for r in all_results.values())
    col_width = 90 if has_hallucination else 70
    print("\n" + "=" * col_width)
    header = f"{'Variant':<8} {'Hit Rate':>10} {'Pos Error':>10} {'Terminal Drift':>15} {'S1%':>6} {'S2%':>6}"
    if has_hallucination:
        header += f" {'Hallucination':>14}"
    print(header)
    print("-" * col_width)
    for v, r in all_results.items():
        hr = f"{r['mean_waypoint_hit_rate']:.2f}" if r["mean_waypoint_hit_rate"] is not None else "  N/A"
        pe = f"{r['mean_waypoint_position_error']:.3f}" if r["mean_waypoint_position_error"] is not None else "   N/A"
        td = f"{r['mean_terminal_drift']:.4f}"
        fc = r["failure_classification"]
        s1p = f"{fc['S1_fraction']:.0%}"
        s2p = f"{fc['S2_fraction']:.0%}"
        row = f"{v:<8} {hr:>10} {pe:>10} {td:>15} {s1p:>6} {s2p:>6}"
        if has_hallucination:
            hs = r["mean_hallucination_score"]
            row += f" {f'{hs:.3f}':>14}" if hs is not None else f" {'  N/A':>14}"
        print(row)
    print("=" * col_width)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
