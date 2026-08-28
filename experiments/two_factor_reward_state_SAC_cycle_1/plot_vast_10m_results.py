"""Plot cross-arm probe summaries from the 10M-step Vast campaign."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
FIGURES_DIR = STUDY_DIR / "figures"
CONDITIONS = ("reward_both", "reward_factor_1", "reward_factor_2")
CONDITION_LABELS = {
    "reward_both": "reward both",
    "reward_factor_1": "reward factor 1",
    "reward_factor_2": "reward factor 2",
}
TARGETS = (
    ("joint_mixed_state", "joint"),
    ("factor_1", "factor 1"),
    ("factor_2", "factor 2"),
)
COLORS = {
    "joint_mixed_state": "#355c9a",
    "factor_1": "#c45135",
    "factor_2": "#2a9d8f",
}


def _latest_run_dir(condition: str) -> Path:
    results_root = STUDY_DIR / condition / "results"
    run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"no results under {results_root}")
    return run_dirs[-1]


def _load_summaries() -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for condition in CONDITIONS:
        run_dir = _latest_run_dir(condition)
        summaries[condition] = json.loads(
            (run_dir / "condition_summary.json").read_text()
        )
    return summaries


def _final_report(summary: dict) -> dict:
    return max(summary["checkpoint_reports"], key=lambda row: row["agent_steps"])


def plot_final_r2_bars(summaries: dict[str, dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(CONDITIONS))
    width = 0.24
    for index, (target_key, target_label) in enumerate(TARGETS):
        values = [
            _final_report(summaries[condition])["probe_fits"][target_key]["r_squared"]
            for condition in CONDITIONS
        ]
        offset = (index - 1) * width
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=target_label,
            color=COLORS[target_key],
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x)
    axis.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    axis.set_ylabel("Held-out linear-probe R²")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Final checkpoint probes at 10M steps (seed 42)")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_final_mse_bars(summaries: dict[str, dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(CONDITIONS))
    width = 0.24
    for index, (target_key, target_label) in enumerate(TARGETS):
        values = [
            _final_report(summaries[condition])["probe_fits"][target_key]["mse"]
            for condition in CONDITIONS
        ]
        offset = (index - 1) * width
        axis.bar(
            x + offset,
            values,
            width,
            label=target_label,
            color=COLORS[target_key],
        )
    axis.set_xticks(x)
    axis.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    axis.set_ylabel("Held-out linear-probe MSE")
    axis.set_title("Final checkpoint probe MSE at 10M steps (seed 42)")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_r2_trajectories(summaries: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for axis, condition in zip(axes, CONDITIONS, strict=True):
        reports = summaries[condition]["checkpoint_reports"]
        steps = np.asarray([row["agent_steps"] for row in reports])
        for target_key, target_label in TARGETS:
            r2 = [
                row["probe_fits"][target_key]["r_squared"] for row in reports
            ]
            axis.plot(steps, r2, marker="o", ms=3, label=target_label, color=COLORS[target_key])
        axis.set_title(CONDITION_LABELS[condition])
        axis.set_xlabel("Environment steps")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out linear-probe R²")
    axes[-1].legend(loc="lower right", fontsize=8)
    figure.suptitle("Probe R² trajectories (log-spaced checkpoints)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_target_comparison(summaries: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    condition_colors = {
        "reward_both": "#264653",
        "reward_factor_1": "#e76f51",
        "reward_factor_2": "#2a9d8f",
    }
    for axis, (target_key, target_label) in zip(axes, TARGETS, strict=True):
        for condition in CONDITIONS:
            reports = summaries[condition]["checkpoint_reports"]
            steps = np.asarray([row["agent_steps"] for row in reports])
            r2 = [row["probe_fits"][target_key]["r_squared"] for row in reports]
            axis.plot(
                steps,
                r2,
                marker="o",
                ms=3,
                label=CONDITION_LABELS[condition],
                color=condition_colors[condition],
            )
        axis.set_title(target_label)
        axis.set_xlabel("Environment steps")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out linear-probe R²")
    axes[-1].legend(loc="lower right", fontsize=8)
    figure.suptitle("Reward-arm comparison by probe target")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_cev95_trajectories(summaries: dict[str, dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    condition_colors = {
        "reward_both": "#264653",
        "reward_factor_1": "#e76f51",
        "reward_factor_2": "#2a9d8f",
    }
    for condition in CONDITIONS:
        reports = summaries[condition]["checkpoint_reports"]
        steps = np.asarray([row["agent_steps"] for row in reports])
        dims = np.asarray([row["actor_cev95_dimension"] for row in reports])
        axis.plot(
            steps,
            dims,
            marker="o",
            ms=3,
            label=CONDITION_LABELS[condition],
            color=condition_colors[condition],
        )
    axis.set_xlabel("Environment steps")
    axis.set_xscale("log")
    axis.set_ylabel("Actor dimensions for 95% CEV")
    axis.set_title("Actor effective dimension over training")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    summaries = _load_summaries()
    plot_final_r2_bars(summaries, FIGURES_DIR / "final_probe_r2_bars.png")
    plot_final_mse_bars(summaries, FIGURES_DIR / "final_probe_mse_bars.png")
    plot_r2_trajectories(summaries, FIGURES_DIR / "probe_r2_trajectories_by_arm.png")
    plot_target_comparison(summaries, FIGURES_DIR / "probe_r2_trajectories_by_target.png")
    plot_cev95_trajectories(summaries, FIGURES_DIR / "actor_cev95_trajectories.png")
    print(f"Wrote figures under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
