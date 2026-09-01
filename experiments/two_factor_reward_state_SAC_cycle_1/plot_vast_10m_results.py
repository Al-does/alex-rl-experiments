"""Plot cross-arm probe summaries from the 10M-step Vast campaign."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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
ACCURACY_COLOR = "#7d3ac1"
ACCURACY_LABEL_COLOR = "#5a2d91"
ACCURACY_PANELS = ("reward_both", "reward_factor_1")


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


def _load_probe_reports(condition: str) -> list[dict[str, object]]:
    run_dir = _latest_run_dir(condition)
    reports = [
        json.loads(path.read_text())
        for path in sorted(
            run_dir.glob("checkpoint_probes/steps_*/probe_battery.json")
        )
    ]
    if not reports:
        raise FileNotFoundError(
            f"no checkpoint probe batteries under {run_dir / 'checkpoint_probes'}"
        )
    return sorted(reports, key=lambda row: int(row["agent_steps"]))


def _task_accuracy(report: dict[str, object], condition: str) -> float:
    policy = report["policy"]
    if condition == "reward_both":
        return float(policy["mean_reward"])
    if condition == "reward_factor_1":
        return float(policy["factor_1_state_2_fraction"])
    return float(policy["factor_2_state_2_fraction"])


def _bayes_max(summary: dict[str, object], condition: str) -> float:
    qmdp = float(summary["demand_audit"]["qmdp"])
    if condition == "reward_both":
        return 2.0 * qmdp
    return qmdp


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


def plot_mse_trajectories(summaries: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for axis, condition in zip(axes, CONDITIONS, strict=True):
        reports = summaries[condition]["checkpoint_reports"]
        steps = np.asarray([row["agent_steps"] for row in reports])
        for target_key, target_label in TARGETS:
            mse = [row["probe_fits"][target_key]["mse"] for row in reports]
            axis.plot(
                steps,
                mse,
                marker="o",
                ms=3,
                label=target_label,
                color=COLORS[target_key],
            )
        axis.set_title(CONDITION_LABELS[condition])
        axis.set_xlabel("Environment steps")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out linear-probe MSE")
    axes[-1].legend(loc="upper right", fontsize=8)
    figure.suptitle("Probe MSE trajectories (log-spaced checkpoints)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_mse_trajectories_with_accuracy(
    summaries: dict[str, dict],
    path: Path,
    *,
    log_x: bool = True,
    log_y: bool = False,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), sharey=False)
    legend_handles: list[object] | None = None

    for axis, condition in zip(axes, ACCURACY_PANELS, strict=True):
        summary = summaries[condition]
        probe_reports = _load_probe_reports(condition)
        steps = np.asarray([int(row["agent_steps"]) for row in probe_reports])
        bayes_max = _bayes_max(summary, condition)
        accuracy = np.asarray(
            [_task_accuracy(row, condition) for row in probe_reports],
            dtype=np.float64,
        )
        mse_lines = []
        for target_key, target_label in TARGETS:
            mse = [
                row["probe_fits"][target_key]["mse"] for row in probe_reports
            ]
            mse_lines.append(
                axis.plot(
                    steps,
                    mse,
                    marker="o",
                    ms=3,
                    label=target_label,
                    color=COLORS[target_key],
                )[0]
            )
        axis.set_title(CONDITION_LABELS[condition])
        axis.set_xlabel("Environment steps")
        if log_x:
            axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_ylabel("Held-out linear-probe MSE")
        axis.grid(alpha=0.2)

        accuracy_axis = axis.twinx()
        accuracy_line = accuracy_axis.plot(
            steps,
            accuracy,
            color=ACCURACY_COLOR,
            linewidth=1.9,
            marker="s",
            ms=3.5,
            label="greedy task accuracy",
            zorder=4,
        )[0]
        accuracy_axis.set_ylim(0.0, bayes_max)
        accuracy_axis.set_yticks(np.linspace(0.0, bayes_max, 5))
        accuracy_axis.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=1)
        )
        accuracy_axis.set_ylabel(
            f"Task accuracy (Bayes max {bayes_max:.1%})",
            color=ACCURACY_LABEL_COLOR,
        )
        accuracy_axis.tick_params(axis="y", colors=ACCURACY_LABEL_COLOR)
        accuracy_axis.spines["right"].set_color(ACCURACY_COLOR)

        if legend_handles is None:
            legend_handles = [*mse_lines, accuracy_line]

    if legend_handles is None:
        raise RuntimeError("expected at least one panel while plotting accuracy")
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        fontsize=8,
        frameon=True,
    )
    figure.suptitle(
        "Probe MSE and greedy task accuracy over training (10M steps, seed 42)",
        y=1.16,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
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
    plot_mse_trajectories(summaries, FIGURES_DIR / "probe_mse_trajectories_by_arm.png")
    plot_mse_trajectories_with_accuracy(
        summaries,
        FIGURES_DIR / "probe_mse_trajectories_with_accuracy_by_arm.png",
        log_x=True,
    )
    plot_mse_trajectories_with_accuracy(
        summaries,
        FIGURES_DIR / "probe_mse_trajectories_with_accuracy_by_arm_linear_x.png",
        log_x=False,
    )
    plot_mse_trajectories_with_accuracy(
        summaries,
        FIGURES_DIR / "probe_mse_trajectories_with_accuracy_by_arm_linear_x_log_y.png",
        log_x=False,
        log_y=True,
    )
    plot_r2_trajectories(summaries, FIGURES_DIR / "probe_r2_trajectories_by_arm.png")
    plot_target_comparison(summaries, FIGURES_DIR / "probe_r2_trajectories_by_target.png")
    plot_cev95_trajectories(summaries, FIGURES_DIR / "actor_cev95_trajectories.png")
    print(f"Wrote figures under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
