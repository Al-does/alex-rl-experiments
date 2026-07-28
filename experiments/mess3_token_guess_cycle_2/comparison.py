"""Aggregate completed token-guess cycle-2 seeds into one comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from experiments.mess3_token_guess_cycle_2.shared import CONDITIONS

METRIC_KEYS = (
    "r_squared",
    "token_accuracy_greedy",
    "mse",
    "global_mse_ratio",
    "fine_mse_ratio",
)


def _final_probe(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    probe = summary.get("final_probe")
    if isinstance(probe, Mapping):
        return probe
    raise ValueError("condition summary missing final_probe")


def summarize_seeds(
    summaries: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    missing = {condition.name for condition in CONDITIONS} - set(summaries)
    if missing:
        raise ValueError(f"missing cycle-2 summaries: {sorted(missing)}")
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        runs = sorted(summaries[condition.name], key=lambda run: int(run["seed"]))
        if not runs:
            raise ValueError(f"{condition.name} has no completed seeds")
        metrics: dict[str, Any] = {}
        for key in METRIC_KEYS:
            values = [
                float(_final_probe(run)[key])
                for run in runs
                if _final_probe(run).get(key) is not None
            ]
            metrics[key] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
                if values
                else None
            )
        conditions[condition.name] = {
            "algorithm": condition.algorithm,
            "objective": condition.objective,
            "seeds": [int(run["seed"]) for run in runs],
            "metrics": metrics,
            "task_success_improved": [
                bool(run.get("training_change", {}).get("task_success_improved"))
                for run in runs
            ],
        }
    return {
        "gamma": 0.0,
        "lambda": 0.0,
        "lr": 1e-4,
        "predictive_loss_weight": 1.0,
        "direct_kelly_loss_weight": 1.0,
        "conditions": conditions,
    }


def _plot(summary: Mapping[str, Any], *, path: Path) -> None:
    conditions = summary["conditions"]
    names = [condition.name for condition in CONDITIONS]
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
    _plot(summary, path=output_dir / "token_guess_cycle_2_comparison.png")
    lines = [
        "# MESS3 token-guess cycle 2",
        "",
        "LR=1e-4, predictive coeff=1.0, kelly coeff=1.0, gamma=0.",
        "",
        "| condition | seeds | belief R² | token accuracy | mse |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = summary["conditions"][condition.name]
        metrics = values["metrics"]
        lines.append(
            f"| {condition.name} | {len(values['seeds'])} | "
            f"{metrics['r_squared']['mean']:.4f} ± "
            f"{metrics['r_squared']['std']:.4f} | "
            f"{metrics['token_accuracy_greedy']['mean']:.4f} ± "
            f"{metrics['token_accuracy_greedy']['std']:.4f} | "
            f"{metrics['mse']['mean']:.6f} ± "
            f"{metrics['mse']['std']:.6f} |"
        )
    lines.append("")
    (output_dir / "findings.md").write_text("\n".join(lines))
    return summary
