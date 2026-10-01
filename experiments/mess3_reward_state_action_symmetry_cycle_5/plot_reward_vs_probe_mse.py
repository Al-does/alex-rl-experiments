"""Plot belief-probe MSE and greedy reward occupancy over training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from experiments.mess3_reward_state_action_symmetry_cycle_5.design import (
    analytic_design_summary,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.plot_variant_mse_curves import (
    load_curves,
)

DEFAULT_MAX_STEPS = 760_000


def bayes_max_reward(variant: int) -> float:
    """Return the fully observed oracle state-2 occupancy for one variant."""

    summary = analytic_design_summary()["fully_observed"][f"variant_{variant}"]
    return float(summary["oracle_stationary_state_2"])


def load_reward_mse_trajectory(
    results_dir: Path,
    *,
    variant: int,
    seeds: list[int],
    max_agent_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    curves = load_curves(results_dir, variant, seeds)
    by_index: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for seed, points in curves.items():
        for checkpoint_index, point in enumerate(points):
            steps = int(point["agent_steps"])
            if steps >= max_agent_steps:
                continue
            by_index[checkpoint_index].append(
                {
                    "seed": seed,
                    "checkpoint_index": checkpoint_index,
                    "agent_steps": steps,
                    "mse": float(point["mse"]),
                    "reward": float(point["reward_state_2_fraction_greedy"]),
                }
            )

    trajectory_points = []
    for checkpoint_index in sorted(by_index):
        rows = by_index[checkpoint_index]
        steps = np.asarray([row["agent_steps"] for row in rows], dtype=np.float64)
        mse = np.asarray([row["mse"] for row in rows], dtype=np.float64)
        reward = np.asarray([row["reward"] for row in rows], dtype=np.float64)
        trajectory_points.append(
            {
                "checkpoint_index": checkpoint_index,
                "agent_steps_mean": float(steps.mean()),
                "agent_steps_min": int(steps.min()),
                "agent_steps_max": int(steps.max()),
                "mse_mean": float(mse.mean()),
                "mse_sd": float(mse.std(ddof=0)),
                "reward_mean": float(reward.mean()),
                "reward_sd": float(reward.std(ddof=0)),
                "n_seeds": len(rows),
            }
        )

    if not trajectory_points:
        raise ValueError(f"no checkpoints below {max_agent_steps} for variant {variant}")

    return {
        "variant": variant,
        "seeds": list(seeds),
        "max_agent_steps": max_agent_steps,
        "bayes_max_reward": bayes_max_reward(variant),
        "points": trajectory_points,
    }


def plot_reward_mse_trajectory(
    trajectory: dict[str, Any],
    *,
    title: str,
    output_path: Path,
) -> Path:
    points = trajectory["points"]
    steps = np.asarray([point["agent_steps_mean"] for point in points])
    mse_mean = np.asarray([point["mse_mean"] for point in points])
    mse_sd = np.asarray([point["mse_sd"] for point in points])
    reward_mean = np.asarray([point["reward_mean"] for point in points])
    reward_sd = np.asarray([point["reward_sd"] for point in points])
    bayes_max = float(trajectory["bayes_max_reward"])

    figure, left = plt.subplots(figsize=(8.4, 4.8))
    right = left.twinx()

    mse_color = "#1768ac"
    reward_color = "#dc7c17"

    lower = np.maximum(mse_mean - mse_sd, np.finfo(float).tiny)
    upper = mse_mean + mse_sd
    left.fill_between(steps, lower, upper, color=mse_color, alpha=0.12, linewidth=0)
    left.plot(
        steps,
        mse_mean,
        color=mse_color,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Probe MSE",
    )

    right.fill_between(
        steps,
        reward_mean - reward_sd,
        reward_mean + reward_sd,
        color=reward_color,
        alpha=0.12,
        linewidth=0,
    )
    right.plot(
        steps,
        reward_mean,
        color=reward_color,
        marker="^",
        markersize=4.5,
        linewidth=1.9,
        label="Greedy reward-state-2 occupancy",
    )

    left.set_yscale("log")
    left.set_xlabel("Environment steps")
    left.set_ylabel("Held-out affine-probe MSE (log scale)", color="#333333")
    right.set_ylabel(
        f"Reward occupancy (Bayes max {bayes_max:.1%})",
        color="#a85d0b",
    )
    right.set_ylim(0.0, bayes_max)
    right.set_yticks(np.linspace(0.0, bayes_max, 5))
    right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    right.tick_params(axis="y", colors="#a85d0b")
    right.spines["right"].set_color(reward_color)

    step_formatter = FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
    )
    left.xaxis.set_major_formatter(step_formatter)
    left.set_xlim(left=0.0, right=float(steps.max()) * 1.02)
    left.grid(alpha=0.22, which="both")
    left.set_title(title)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_reward_vs_probe_mse_scatter(
    results_dir: Path,
    *,
    variant: int,
    seeds: list[int],
    max_agent_steps: int,
    title: str,
    output_path: Path,
) -> Path:
    """Scatter of reward against probe MSE (one trajectory per seed)."""

    curves = load_curves(results_dir, variant, seeds)
    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

    mean_mse: list[float] = []
    mean_reward: list[float] = []
    std_mse: list[float] = []
    std_reward: list[float] = []

    filtered = {
        seed: [point for point in points if int(point["agent_steps"]) < max_agent_steps]
        for seed, points in curves.items()
    }
    n_checkpoints = len(next(iter(filtered.values())))
    for checkpoint_index in range(n_checkpoints):
        mse_values = [filtered[seed][checkpoint_index]["mse"] for seed in seeds]
        reward_values = [
            filtered[seed][checkpoint_index]["reward_state_2_fraction_greedy"]
            for seed in seeds
        ]
        mean_mse.append(float(np.mean(mse_values)))
        mean_reward.append(float(np.mean(reward_values)))
        std_mse.append(float(np.std(mse_values)))
        std_reward.append(float(np.std(reward_values)))

    for color, seed in zip(colors, seeds, strict=True):
        rows = filtered[seed]
        mse = np.asarray([row["mse"] for row in rows], dtype=np.float64)
        reward = np.asarray(
            [row["reward_state_2_fraction_greedy"] for row in rows],
            dtype=np.float64,
        )
        axis.plot(mse, reward, color=color, alpha=0.45, linewidth=1.2)
        axis.scatter(
            mse,
            reward,
            color=color,
            s=28,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=f"seed {seed}",
        )

    mean_mse_arr = np.asarray(mean_mse)
    mean_reward_arr = np.asarray(mean_reward)
    axis.plot(
        mean_mse_arr,
        mean_reward_arr,
        color="black",
        linewidth=2.4,
        marker="s",
        markersize=5,
        label=f"mean ({len(seeds)} seeds)",
    )
    axis.errorbar(
        mean_mse_arr,
        mean_reward_arr,
        xerr=np.asarray(std_mse),
        yerr=np.asarray(std_reward),
        fmt="none",
        ecolor="black",
        alpha=0.35,
        capsize=2.5,
    )
    axis.set_xscale("log")
    axis.set_xlabel("Held-out affine-probe MSE (lower is better)")
    axis.set_ylabel("Greedy reward-state-2 occupancy")
    axis.set_title(title)
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Include checkpoints with agent_steps strictly below this value.",
    )
    parser.add_argument(
        "--layout",
        choices=("trajectory", "scatter"),
        default="trajectory",
        help="trajectory = MSE and reward vs training steps; scatter = reward vs MSE.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (defaults under variant_N/figures/).",
    )
    args = parser.parse_args()

    results_dir = args.study_root / f"variant_{args.variant}" / "results"
    trajectory = load_reward_mse_trajectory(
        results_dir,
        variant=args.variant,
        seeds=args.seeds,
        max_agent_steps=args.max_agent_steps,
    )
    last_steps = trajectory["points"][-1]["agent_steps_mean"]
    default_stem = (
        f"variant_{args.variant}_reward_mse_trajectory_lt_{args.max_agent_steps}"
        if args.layout == "trajectory"
        else f"variant_{args.variant}_reward_vs_probe_mse_lt_{args.max_agent_steps}"
    )
    output = args.output or (
        args.study_root / f"variant_{args.variant}" / "figures" / f"{default_stem}.png"
    )
    title = (
        f"Cycle 5 variant {args.variant} through {last_steps / 1_000_000:.2f}M steps "
        f"(n={len(args.seeds)} seeds)"
    )
    if args.layout == "trajectory":
        print(plot_reward_mse_trajectory(trajectory, title=title, output_path=output))
    else:
        print(
            plot_reward_vs_probe_mse_scatter(
                results_dir,
                variant=args.variant,
                seeds=args.seeds,
                max_agent_steps=args.max_agent_steps,
                title=title,
                output_path=output,
            )
        )


if __name__ == "__main__":
    main()
