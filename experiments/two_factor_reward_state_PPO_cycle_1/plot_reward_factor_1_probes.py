"""Plot factor probe MSE and task occupancy for reward_factor_1 PPO seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from experiments.two_factor_reward_state_PPO_cycle_1.reference import (
    bayes_max_reward_factor_1,
)

STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_ROOT / "reward_factor_1" / "results"
DEFAULT_OUTPUT = (
    DEFAULT_RESULTS_ROOT / "reward_factor_1_probe_mse_and_accuracy.png"
)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
# First log-spaced checkpoint strictly after 2,000,000 environment steps.
DEFAULT_MAX_CHECKPOINT_INDEX = 10


def _load_seed_reports(results_root: Path, seed: int) -> list[dict[str, Any]]:
    run_dir = (
        results_root
        / f"two_factor_reward_state_PPO_cycle_1-reward_factor_1-seed{seed}"
    )
    pattern = "checkpoint_probes/steps_*/probe_battery.json"
    reports = []
    for path in sorted(run_dir.glob(pattern)):
        report = json.loads(path.read_text())
        reports.append(report)
    if len(reports) != 14:
        raise FileNotFoundError(
            f"expected 14 checkpoint probes for seed {seed}, found {len(reports)}"
        )
    return sorted(reports, key=lambda row: int(row["agent_steps"]))


def load_trajectory(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_checkpoint_index: int | None = DEFAULT_MAX_CHECKPOINT_INDEX,
) -> dict[str, Any]:
    """Aggregate checkpoint-aligned probe trajectories across seeds."""

    by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        for checkpoint_index, report in enumerate(_load_seed_reports(results_root, seed)):
            if (
                max_checkpoint_index is not None
                and checkpoint_index > max_checkpoint_index
            ):
                continue
            by_index[checkpoint_index].append(
                {
                    "seed": seed,
                    "checkpoint_index": checkpoint_index,
                    "agent_steps": int(report["agent_steps"]),
                    "checkpoint": report["checkpoint"],
                    "factor_1_mse": float(report["probe_fits"]["factor_1"]["mse"]),
                    "factor_2_mse": float(report["probe_fits"]["factor_2"]["mse"]),
                    "task_accuracy": float(report["policy"]["mean_reward"]),
                }
            )

    points = []
    for checkpoint_index in sorted(by_index):
        rows = by_index[checkpoint_index]
        steps = np.asarray([row["agent_steps"] for row in rows], dtype=np.float64)
        factor_1 = np.asarray(
            [row["factor_1_mse"] for row in rows], dtype=np.float64
        )
        factor_2 = np.asarray(
            [row["factor_2_mse"] for row in rows], dtype=np.float64
        )
        accuracy = np.asarray(
            [row["task_accuracy"] for row in rows], dtype=np.float64
        )
        points.append(
            {
                "checkpoint_index": checkpoint_index,
                "checkpoint": rows[0]["checkpoint"],
                "agent_steps_mean": float(steps.mean()),
                "agent_steps_min": int(steps.min()),
                "agent_steps_max": int(steps.max()),
                "factor_1_mse_mean": float(factor_1.mean()),
                "factor_1_mse_sd": float(factor_1.std(ddof=0)),
                "factor_2_mse_mean": float(factor_2.mean()),
                "factor_2_mse_sd": float(factor_2.std(ddof=0)),
                "task_accuracy_mean": float(accuracy.mean()),
                "task_accuracy_sd": float(accuracy.std(ddof=0)),
                "n_seeds": len(rows),
            }
        )
    return {
        "seeds": list(seeds),
        "max_checkpoint_index": max_checkpoint_index,
        "bayes_max_reward": bayes_max_reward_factor_1(),
        "points": points,
    }


def plot_reward_factor_1_probes(
    output: Path = DEFAULT_OUTPUT,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_checkpoint_index: int | None = DEFAULT_MAX_CHECKPOINT_INDEX,
) -> Path:
    """Write dual-axis probe MSE and greedy occupancy trajectory."""

    trajectory = load_trajectory(
        results_root=results_root,
        seeds=seeds,
        max_checkpoint_index=max_checkpoint_index,
    )
    points = trajectory["points"]
    if not points:
        raise ValueError("no checkpoint probes matched the requested step window")
    steps = np.asarray([point["agent_steps_mean"] for point in points])
    factor_1_mean = np.asarray([point["factor_1_mse_mean"] for point in points])
    factor_1_sd = np.asarray([point["factor_1_mse_sd"] for point in points])
    factor_2_mean = np.asarray([point["factor_2_mse_mean"] for point in points])
    factor_2_sd = np.asarray([point["factor_2_mse_sd"] for point in points])
    accuracy_mean = np.asarray([point["task_accuracy_mean"] for point in points])
    accuracy_sd = np.asarray([point["task_accuracy_sd"] for point in points])
    bayes_max = float(trajectory["bayes_max_reward"])

    figure, left = plt.subplots(figsize=(8.4, 4.8))
    right = left.twinx()

    factor_1_color = "#1768ac"
    factor_2_color = "#2d7d46"
    accuracy_color = "#dc7c17"

    for mean, sd, color in (
        (factor_1_mean, factor_1_sd, factor_1_color),
        (factor_2_mean, factor_2_sd, factor_2_color),
    ):
        lower = np.maximum(mean - sd, np.finfo(float).tiny)
        upper = mean + sd
        left.fill_between(steps, lower, upper, color=color, alpha=0.12, linewidth=0)

    left.plot(
        steps,
        factor_1_mean,
        color=factor_1_color,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Factor 1 probe MSE",
    )
    left.plot(
        steps,
        factor_2_mean,
        color=factor_2_color,
        marker="s",
        markersize=4.0,
        linewidth=2.0,
        label="Factor 2 probe MSE",
    )

    right.fill_between(
        steps,
        accuracy_mean - accuracy_sd,
        accuracy_mean + accuracy_sd,
        color=accuracy_color,
        alpha=0.12,
        linewidth=0,
    )
    right.plot(
        steps,
        accuracy_mean,
        color=accuracy_color,
        marker="^",
        markersize=4.5,
        linewidth=1.9,
        label="Greedy task accuracy",
    )

    left.set_yscale("log")
    left.set_xlabel("Environment steps")
    left.set_ylabel("Held-out linear-probe MSE (log scale)", color="#333333")
    right.set_ylabel(
        f"Task accuracy (Bayes max {bayes_max:.1%})",
        color="#a85d0b",
    )
    right.set_ylim(0.0, bayes_max)
    right.set_yticks(np.linspace(0.0, bayes_max, 5))
    right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    right.tick_params(axis="y", colors="#a85d0b")
    right.spines["right"].set_color(accuracy_color)

    step_formatter = FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
    )
    left.xaxis.set_major_formatter(step_formatter)
    left.set_xlim(left=0.0, right=float(steps.max()) * 1.02)
    left.grid(alpha=0.22, which="both")
    last = points[-1]
    left.set_title(
        "reward_factor_1 PPO through "
        f"{last['agent_steps_mean'] / 1_000_000:.2f}M steps "
        f"(seeds {min(seeds)}–{max(seeds)}, n={len(seeds)})"
    )

    left_handles, left_labels = left.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    left.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper right",
        fontsize=8.5,
        frameon=False,
    )

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--max-checkpoint-index",
        type=int,
        default=DEFAULT_MAX_CHECKPOINT_INDEX,
        help=(
            "Include checkpoints up to this index (10 is the first checkpoint "
            "just north of 2M steps)."
        ),
    )
    args = parser.parse_args(argv)
    path = plot_reward_factor_1_probes(
        args.output,
        results_root=args.results_root,
        seeds=tuple(args.seeds),
        max_checkpoint_index=args.max_checkpoint_index,
    )
    print(path)


if __name__ == "__main__":
    main()
