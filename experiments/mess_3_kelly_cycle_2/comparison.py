"""Aggregate completed cycle-2 seeds and critic pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess_3_kelly_cycle_2.shared import ARMS


ACTOR_MODES = (
    "correctness",
    "coupled_kelly",
    "decoupled_kelly",
    "conditional_decoupled_kelly",
)


def summarize_seeds(
    summaries: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate one or more completed seed summaries per arm."""

    missing = {arm.name for arm in ARMS} - set(summaries)
    if missing:
        raise ValueError(f"missing cycle-2 arm summaries: {sorted(missing)}")
    conditions: dict[str, Any] = {}
    for arm in ARMS:
        runs = summaries[arm.name]
        if not runs:
            raise ValueError(f"{arm.name} has no completed seeds")
        metrics: dict[str, Any] = {}
        for key in (
            "r_squared",
            "token_accuracy_greedy",
            "expected_log_growth_mean",
            "wager_vs_oracle_rmse",
        ):
            values = [
                float(run["probe"][key])
                for run in runs
                if run["probe"][key] is not None
            ]
            metrics[key] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
                if values
                else None
            )
        conditions[arm.name] = {
            "actor_mode": arm.actor_mode,
            "critic_mode": arm.critic_mode,
            "wager_layout": arm.wager_layout,
            "seeds": [int(run["seed"]) for run in runs],
            "metrics": metrics,
            "any_wager_collapse": any(
                bool(run["wager_collapse_detected"]) for run in runs
            ),
        }
    return {"conditions": conditions}


def _plot(summary: Mapping[str, Any], *, path: Path) -> None:
    conditions = summary["conditions"]
    x = np.arange(len(ACTOR_MODES))
    width = 0.36
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)
    for offset, critic, color in (
        (-width / 2, "mean", "tab:blue"),
        (width / 2, "iqn", "tab:orange"),
    ):
        names = [f"{actor}_{critic}" for actor in ACTOR_MODES]
        r2 = [
            conditions[name]["metrics"]["r_squared"]["mean"] for name in names
        ]
        accuracy = [
            conditions[name]["metrics"]["token_accuracy_greedy"]["mean"]
            for name in names
        ]
        axes[0].bar(x + offset, r2, width, label=critic, color=color)
        axes[1].bar(x + offset, accuracy, width, label=critic, color=color)
    axes[0].set_ylabel("held-out belief R²")
    axes[1].set_ylabel("greedy token accuracy")
    axes[1].set_xticks(
        x,
        [name.replace("_", "\n") for name in ACTOR_MODES],
    )
    for axis in axes:
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_comparison(
    summaries: Mapping[str, list[Mapping[str, Any]]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Write compact JSON, Markdown, and a paired mean/IQN plot."""

    summary = summarize_seeds(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _plot(summary, path=output_dir / "cycle_2_comparison.png")
    lines = [
        "# MESS3 Kelly cycle 2",
        "",
        "| condition | seeds | belief R² | token accuracy | expected growth | collapse |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = summary["conditions"][arm.name]
        metrics = values["metrics"]
        growth = metrics["expected_log_growth_mean"]
        growth_text = "—" if growth is None else f"{growth['mean']:.6f}"
        lines.append(
            f"| {arm.name} | {len(values['seeds'])} | "
            f"{metrics['r_squared']['mean']:.4f} ± "
            f"{metrics['r_squared']['std']:.4f} | "
            f"{metrics['token_accuracy_greedy']['mean']:.4f} ± "
            f"{metrics['token_accuracy_greedy']['std']:.4f} | "
            f"{growth_text} | "
            f"{str(values['any_wager_collapse']).lower()} |"
        )
    lines.append("")
    (output_dir / "findings.md").write_text("\n".join(lines))
    return summary
