"""Dual-axis probe MSE and reward-occupancy trajectories for cycle-2 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

FACTOR_MSE_COLORS = {
    "factor_1": "#c45135",
    "factor_2": "#2a9d8f",
}
OCCUPANCY_COLOR = "#7d3ac1"
OCCUPANCY_LABEL_COLOR = "#5a2d91"

EXPERIMENTS = (
    ("SAC", "reward_both"),
    ("SAC", "reward_factor_1"),
    ("PPO", "reward_both"),
    ("PPO", "reward_factor_1"),
)
CONDITION_LABELS = {
    "reward_both": "reward both factors",
    "reward_factor_1": "reward factor 1 only",
}


def _run_dir(algo: str, condition: str, seed: int) -> Path:
    results_root = (
        STUDY_ROOT
        / f"two_factor_reward_state_{algo}_cycle_2"
        / condition
        / "results"
    )
    matches = sorted(results_root.glob(f"*-seed{seed}"))
    if not matches:
        raise FileNotFoundError(f"no run for {algo} {condition} seed {seed}")
    return matches[0]


def _load_probe_reports(algo: str, condition: str, seed: int) -> list[dict]:
    run_dir = _run_dir(algo, condition, seed)
    reports = [
        json.loads(path.read_text())
        for path in sorted(run_dir.glob("checkpoint_probes/steps_*/probe_battery.json"))
    ]
    if not reports:
        raise FileNotFoundError(f"no probe batteries under {run_dir}")
    return sorted(reports, key=lambda row: int(row["agent_steps"]))


def _reward_occupancy(report: dict, condition: str) -> float:
    policy = report["policy"]
    f1 = float(policy["factor_1_state_2_fraction"])
    f2 = float(policy["factor_2_state_2_fraction"])
    if condition == "reward_both":
        return 0.5 * (f1 + f2)
    return f1


def _reference_schedule(
    curves: dict[int, list[dict]],
) -> tuple[list[int], list[int]]:
    seeds = sorted(curves)
    n_points = len(curves[seeds[0]])
    for seed in seeds:
        points = curves[seed]
        if len(points) != n_points:
            raise ValueError(
                f"seed {seed} has {len(points)} checkpoints, expected {n_points}"
            )
    middle_seed = seeds[len(seeds) // 2]
    schedule = [int(point["agent_steps"]) for point in curves[middle_seed]]
    return schedule, seeds


def load_experiment_curves(
    algo: str,
    condition: str,
    seeds: tuple[int, ...],
) -> dict[int, list[dict]]:
    curves: dict[int, list[dict]] = {}
    for seed in seeds:
        curves[seed] = _load_probe_reports(algo, condition, seed)
    return curves


def plot_mse_occupancy_trajectory(
    curves: dict[int, list[dict]],
    *,
    algo: str,
    condition: str,
    output_stem: Path,
    seeds: tuple[int, ...],
) -> tuple[Path, Path]:
    schedule, ordered_seeds = _reference_schedule(curves)
    steps = np.asarray(schedule, dtype=np.float64)
    n_points = len(schedule)

    f1_mse = np.full((len(ordered_seeds), n_points), np.nan)
    f2_mse = np.full((len(ordered_seeds), n_points), np.nan)
    occupancy = np.full((len(ordered_seeds), n_points), np.nan)
    for row, seed in enumerate(ordered_seeds):
        for col, report in enumerate(curves[seed]):
            fits = report["probe_fits"]
            f1_mse[row, col] = float(fits["factor_1"]["mse"])
            f2_mse[row, col] = float(fits["factor_2"]["mse"])
            occupancy[row, col] = _reward_occupancy(report, condition)

    f1_mean = np.nanmean(f1_mse, axis=0)
    f1_std = np.nanstd(f1_mse, axis=0)
    f2_mean = np.nanmean(f2_mse, axis=0)
    f2_std = np.nanstd(f2_mse, axis=0)
    occ_mean = np.nanmean(occupancy, axis=0)
    occ_std = np.nanstd(occupancy, axis=0)

    positive = steps[steps > 0]
    if positive.size == 0:
        raise ValueError("expected at least one post-init checkpoint")
    x_max = float(positive.max()) * 1.02

    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    f1_line = axis.plot(
        steps,
        f1_mean,
        color=FACTOR_MSE_COLORS["factor_1"],
        linewidth=2.0,
        marker="o",
        markersize=4.0,
        label="factor 1 probe MSE",
    )[0]
    axis.fill_between(
        steps,
        np.maximum(f1_mean - f1_std, 1e-12),
        f1_mean + f1_std,
        color=FACTOR_MSE_COLORS["factor_1"],
        alpha=0.15,
    )
    f2_line = axis.plot(
        steps,
        f2_mean,
        color=FACTOR_MSE_COLORS["factor_2"],
        linewidth=2.0,
        marker="o",
        markersize=4.0,
        label="factor 2 probe MSE",
    )[0]
    axis.fill_between(
        steps,
        np.maximum(f2_mean - f2_std, 1e-12),
        f2_mean + f2_std,
        color=FACTOR_MSE_COLORS["factor_2"],
        alpha=0.15,
    )

    axis.set_yscale("log")
    axis.set_xlim(-x_max * 0.02, x_max)
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Held-out linear-probe MSE (log scale)")
    axis.grid(alpha=0.25)

    occupancy_axis = axis.twinx()
    occ_line = occupancy_axis.plot(
        steps,
        occ_mean,
        color=OCCUPANCY_COLOR,
        linewidth=2.0,
        marker="s",
        markersize=4.0,
        label="reward occupancy",
        zorder=4,
    )[0]
    occupancy_axis.fill_between(
        steps,
        np.clip(occ_mean - occ_std, 0.0, 1.0),
        np.clip(occ_mean + occ_std, 0.0, 1.0),
        color=OCCUPANCY_COLOR,
        alpha=0.12,
    )
    occupancy_axis.set_ylim(0.0, max(0.45, float(occ_mean.max() + occ_std.max()) * 1.15))
    occupancy_axis.set_ylabel("Reward occupancy (state 2)", color=OCCUPANCY_LABEL_COLOR)
    occupancy_axis.tick_params(axis="y", colors=OCCUPANCY_LABEL_COLOR)
    occupancy_axis.spines["right"].set_color(OCCUPANCY_COLOR)
    occupancy_axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

    step_formatter = FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
    )
    axis.xaxis.set_major_formatter(step_formatter)

    seed_label = ", ".join(str(seed) for seed in seeds)
    axis.set_title(
        f"Cycle 2 {algo} — {CONDITION_LABELS[condition]}\n"
        f"mean ± std over seeds {seed_label}"
    )
    figure.legend(
        handles=[f1_line, f2_line, occ_line],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=9,
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)

    for algo, condition in EXPERIMENTS:
        curves = load_experiment_curves(algo, condition, seeds)
        stem = (
            args.figures_dir
            / f"{algo.lower()}_{condition}_mse_occupancy_trajectory"
        )
        png_path, pdf_path = plot_mse_occupancy_trajectory(
            curves,
            algo=algo,
            condition=condition,
            output_stem=stem,
            seeds=seeds,
        )
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
