"""Plot greedy reward-state occupancy against belief-probe MSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.mess3_reward_state_action_symmetry_cycle_5.plot_variant_mse_curves import (
    load_curves,
)


DEFAULT_MAX_STEPS = 760_000


def load_reward_mse_points(
    results_dir: Path,
    *,
    variant: int,
    seeds: list[int],
    max_agent_steps: int = DEFAULT_MAX_STEPS,
) -> dict[int, list[dict[str, float | int]]]:
    curves = load_curves(results_dir, variant, seeds)
    filtered: dict[int, list[dict[str, float | int]]] = {}
    for seed, points in curves.items():
        rows = []
        for point in points:
            steps = int(point["agent_steps"])
            if steps >= max_agent_steps:
                continue
            rows.append(
                {
                    "agent_steps": steps,
                    "mse": float(point["mse"]),
                    "reward": float(point["reward_state_2_fraction_greedy"]),
                }
            )
        if not rows:
            raise ValueError(f"seed {seed} has no checkpoints below {max_agent_steps}")
        filtered[seed] = rows
    return filtered


def plot_reward_vs_probe_mse(
    points_by_seed: dict[int, list[dict[str, float | int]]],
    *,
    title: str,
    output_path: Path,
    max_agent_steps: int = DEFAULT_MAX_STEPS,
) -> Path:
    seeds = sorted(points_by_seed)
    n_checkpoints = len(points_by_seed[seeds[0]])
    for seed in seeds:
        if len(points_by_seed[seed]) != n_checkpoints:
            raise ValueError("all seeds must share the same filtered checkpoint set")

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

    mean_mse: list[float] = []
    mean_reward: list[float] = []
    std_mse: list[float] = []
    std_reward: list[float] = []

    for checkpoint_index in range(n_checkpoints):
        mse_values = [points_by_seed[seed][checkpoint_index]["mse"] for seed in seeds]
        reward_values = [
            points_by_seed[seed][checkpoint_index]["reward"] for seed in seeds
        ]
        mean_mse.append(float(np.mean(mse_values)))
        mean_reward.append(float(np.mean(reward_values)))
        std_mse.append(float(np.std(mse_values)))
        std_reward.append(float(np.std(reward_values)))

    for color, seed in zip(colors, seeds, strict=True):
        rows = points_by_seed[seed]
        mse = np.asarray([row["mse"] for row in rows], dtype=np.float64)
        reward = np.asarray([row["reward"] for row in rows], dtype=np.float64)
        axis.plot(
            mse,
            reward,
            color=color,
            alpha=0.45,
            linewidth=1.2,
            zorder=2,
        )
        axis.scatter(
            mse,
            reward,
            color=color,
            s=28,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=f"seed {seed}",
            zorder=3,
        )

    mean_mse_arr = np.asarray(mean_mse)
    mean_reward_arr = np.asarray(mean_reward)
    std_mse_arr = np.asarray(std_mse)
    std_reward_arr = np.asarray(std_reward)

    axis.plot(
        mean_mse_arr,
        mean_reward_arr,
        color="black",
        linewidth=2.4,
        marker="s",
        markersize=5,
        label=f"mean ({len(seeds)} seeds)",
        zorder=4,
    )
    axis.errorbar(
        mean_mse_arr,
        mean_reward_arr,
        xerr=std_mse_arr,
        yerr=std_reward_arr,
        fmt="none",
        ecolor="black",
        alpha=0.35,
        capsize=2.5,
        zorder=4,
    )

    for index, (mse, reward) in enumerate(zip(mean_mse_arr, mean_reward_arr, strict=True)):
        steps = int(points_by_seed[seeds[len(seeds) // 2]][index]["agent_steps"])
        if steps == 0:
            label = "init"
        elif index in (0, n_checkpoints - 1) or steps >= max_agent_steps * 0.9:
            label = f"{steps / 1_000_000:.2f}M".rstrip("0").rstrip(".") + "M"
        else:
            continue
        axis.annotate(
            label,
            xy=(mse, reward),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="0.25",
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
        "--output",
        type=Path,
        default=None,
        help="PNG output path (defaults under variant_N/figures/).",
    )
    args = parser.parse_args()

    results_dir = args.study_root / f"variant_{args.variant}" / "results"
    points = load_reward_mse_points(
        results_dir,
        variant=args.variant,
        seeds=args.seeds,
        max_agent_steps=args.max_agent_steps,
    )
    output = args.output or (
        args.study_root
        / f"variant_{args.variant}"
        / "figures"
        / f"variant_{args.variant}_reward_vs_probe_mse_lt_{args.max_agent_steps}.png"
    )
    title = (
        f"Cycle 5 variant {args.variant} — reward vs belief-probe MSE "
        f"(<{args.max_agent_steps / 1_000_000:g}M steps, n={len(args.seeds)} seeds)"
    )
    print(plot_reward_vs_probe_mse(points, title=title, output_path=output))


if __name__ == "__main__":
    main()
