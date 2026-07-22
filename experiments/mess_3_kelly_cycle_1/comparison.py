"""Aggregate independently run Kelly conditions into one compact comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess_3_kelly_cycle_1.shared import CONDITIONS


def write_comparison(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Write a table and figure from one completed summary per condition."""

    missing = set(CONDITIONS) - set(summaries)
    if missing:
        raise ValueError(f"missing Kelly condition summaries: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = {
        condition: {
            "r_squared": float(summaries[condition]["probe"]["r_squared"]),
            "token_accuracy_greedy": float(
                summaries[condition]["probe"]["token_accuracy_greedy"]
            ),
            "wager_mean": float(
                summaries[condition]["probe"]["wager_mean"]
            ),
            "wager_collapse_fraction": float(
                summaries[condition]["probe"]["wager_collapse_fraction"]
            ),
            "expected_log_growth_mean": float(
                summaries[condition]["probe"]["expected_log_growth_mean"]
            ),
            "wager_vs_oracle_rmse": float(
                summaries[condition]["probe"]["wager_vs_oracle_rmse"]
            ),
            "wager_collapse_detected": bool(
                summaries[condition]["wager_collapse_detected"]
            ),
        }
        for condition in CONDITIONS
    }
    summary = {
        "conditions": compact,
        "all_without_warm_start": all(
            not values["warm_start"] for values in summaries.values()
        ),
        "all_without_predictive_auxiliary_loss": all(
            not values["predictive_auxiliary_loss"]
            for values in summaries.values()
        ),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    labels = [condition.replace("_", "\n") for condition in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for axis, key, title in (
        (axes[0, 0], "r_squared", "Held-out rank-2 belief R²"),
        (axes[0, 1], "token_accuracy_greedy", "Greedy token accuracy"),
        (axes[1, 0], "wager_mean", "Mean wager"),
        (
            axes[1, 1],
            "expected_log_growth_mean",
            "Expected log growth per step",
        ),
    ):
        axis.bar(x, [compact[name][key] for name in CONDITIONS])
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(output_dir / "kelly_comparison.png", dpi=220)
    plt.close(figure)

    lines = [
        "# MESS3 Kelly cycle 1",
        "",
        "All four conditions train from scratch without predictive auxiliary loss.",
        "",
        "| condition | belief R² | token accuracy | mean wager | expected log growth | wager RMSE | collapse |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = compact[condition]
        lines.append(
            f"| {condition} | {values['r_squared']:.4f} | "
            f"{values['token_accuracy_greedy']:.4f} | "
            f"{values['wager_mean']:.4f} | "
            f"{values['expected_log_growth_mean']:.6f} | "
            f"{values['wager_vs_oracle_rmse']:.4f} | "
            f"{str(values['wager_collapse_detected']).lower()} |"
        )
    lines.append("")
    (output_dir / "findings.md").write_text("\n".join(lines))
    return summary
