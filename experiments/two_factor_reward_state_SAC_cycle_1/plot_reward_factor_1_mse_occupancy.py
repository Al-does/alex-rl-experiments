"""Dual-axis mean trajectory for reward_factor_1 SAC runs (seeds 42–46)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
CONDITION = "reward_factor_1"
SEEDS = (42, 43, 44, 45, 46)
MAX_STEPS = 2_500_000
EPISODE_LENGTH = 1024
PROBE_TARGET = "factor_1"
FIGURES_DIR = STUDY_DIR / "figures"


def _run_dir(seed: int) -> Path:
    return (
        STUDY_DIR
        / CONDITION
        / "results"
        / f"two_factor_reward_state_SAC_cycle_1-{CONDITION}-seed{seed}"
    )


def _load_probe_trajectories() -> tuple[np.ndarray, np.ndarray]:
    step_rows: list[list[float]] = []
    mse_rows: list[list[float]] = []
    for seed in SEEDS:
        summary = json.loads((_run_dir(seed) / "condition_summary.json").read_text())
        reports = [
            row
            for row in summary["checkpoint_reports"]
            if row["agent_steps"] <= MAX_STEPS
        ]
        step_rows.append([float(row["agent_steps"]) for row in reports])
        mse_rows.append(
            [float(row["probe_fits"][PROBE_TARGET]["mse"]) for row in reports]
        )
    steps = np.mean(np.asarray(step_rows, dtype=np.float64), axis=0)
    mse = np.mean(np.asarray(mse_rows, dtype=np.float64), axis=0)
    return steps, mse


def _load_occupancy_trajectory() -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, MAX_STEPS, 500)
    interpolated: list[np.ndarray] = []
    for seed in SEEDS:
        curve_path = _run_dir(seed) / "training_curves.jsonl"
        steps: list[float] = []
        occupancy: list[float] = []
        for line in curve_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("steps")
            return_mean = row.get("return_mean")
            if step is None or return_mean is None:
                continue
            if float(step) > MAX_STEPS:
                continue
            steps.append(float(step))
            occupancy.append(100.0 * float(return_mean) / EPISODE_LENGTH)
        if not steps:
            raise RuntimeError(f"no occupancy curve for seed {seed}")
        interpolated.append(np.interp(grid, steps, occupancy))
    mean_occupancy = np.mean(np.stack(interpolated, axis=0), axis=0)
    return grid, mean_occupancy


def plot_mean_trajectory(path: Path) -> None:
    probe_steps, probe_mse = _load_probe_trajectories()
    occ_steps, occ_pct = _load_occupancy_trajectory()

    figure, left_axis = plt.subplots(figsize=(8.5, 4.8))
    right_axis = left_axis.twinx()

    left_axis.plot(
        probe_steps,
        probe_mse,
        color="#c45135",
        marker="o",
        ms=5,
        lw=1.8,
        label=f"Mean {PROBE_TARGET.replace('_', ' ')} probe MSE",
    )
    right_axis.plot(
        occ_steps,
        occ_pct,
        color="#2a9d8f",
        lw=2.0,
        alpha=0.9,
        label="Mean factor-1 state-2 occupancy",
    )

    left_axis.set_xlim(0, MAX_STEPS)
    left_axis.set_xlabel("Environment steps")
    left_axis.set_yscale("log")
    left_axis.set_ylabel("Held-out linear-probe MSE (log scale)")
    right_axis.set_ylabel("Reward occupancy (%)")
    right_axis.set_ylim(0.0, 100.0)

    lines_left, labels_left = left_axis.get_legend_handles_labels()
    lines_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="center right",
        fontsize=9,
    )
    left_axis.grid(alpha=0.25)
    left_axis.set_title(
        "reward_factor_1 SAC: mean probe MSE and occupancy (seeds 42–46, 0–2.5M steps)"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    output = FIGURES_DIR / "reward_factor_1_mean_mse_occupancy_2p5m.png"
    plot_mean_trajectory(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
