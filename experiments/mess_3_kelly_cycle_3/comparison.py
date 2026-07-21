"""Merge completed gamma-.99 runs into one compact comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess_3_kelly_cycle_3.shared import ARMS


def summarize_seeds(
    summaries: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    missing = {arm.name for arm in ARMS} - set(summaries)
    if missing:
        raise ValueError(f"missing cycle-3 summaries: {sorted(missing)}")
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
            "critic_mode": arm.critic_mode,
            "conditional_kelly": arm.conditional_kelly,
            "seeds": [int(run["seed"]) for run in runs],
            "metrics": metrics,
            "any_wager_collapse": any(
                bool(run["wager_collapse_detected"]) for run in runs
            ),
        }
    return {"gamma": 0.99, "lambda": 0.95, "conditions": conditions}


def _plot(summary: Mapping[str, Any], *, path: Path) -> None:
    conditions = summary["conditions"]
    names = [arm.name for arm in ARMS]
    labels = [name.replace("_", "\n") for name in names]
    x = np.arange(len(names))
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)
    for axis, key, ylabel in (
        (axes[0], "r_squared", "held-out belief R²"),
        (axes[1], "token_accuracy_greedy", "greedy token accuracy"),
    ):
        means = [conditions[name]["metrics"][key]["mean"] for name in names]
        errors = [conditions[name]["metrics"][key]["std"] for name in names]
        axis.bar(x, means, yerr=errors, capsize=4)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x, labels)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_comparison(
    summaries: Mapping[str, list[Mapping[str, Any]]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    summary = summarize_seeds(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _plot(summary, path=output_dir / "gamma_099_comparison.png")
    lines = [
        "# MESS3 Kelly cycle 3: gamma 0.99",
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
