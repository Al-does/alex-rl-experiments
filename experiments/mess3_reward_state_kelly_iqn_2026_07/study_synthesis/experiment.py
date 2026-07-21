"""Aggregate the eight completed reward-state conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from harness.artifacts import RunArtifacts
from harness.context import RunContext


CONDITIONS = (
    ("ppo_gamma_0", "PPO", 0.0),
    ("ppo_gamma_099", "PPO", 0.99),
    ("iqn_gamma_0", "IQN", 0.0),
    ("iqn_gamma_099", "IQN", 0.99),
    ("kelly_gamma_0", "Kelly", 0.0),
    ("kelly_gamma_099", "Kelly", 0.99),
    ("kelly_iqn_gamma_0", "Kelly + IQN", 0.0),
    ("kelly_iqn_gamma_099", "Kelly + IQN", 0.99),
)


def _latest_summary(family: Path, condition: str, seed: int) -> tuple[Path, dict]:
    candidates = sorted(
        (family / condition / "results").glob("*/condition_summary.json"),
        reverse=True,
    )
    for path in candidates:
        summary = json.loads(path.read_text())
        if not summary.get("smoke", False) and int(summary["seed"]) == seed:
            return path, summary
    raise FileNotFoundError(
        f"no completed non-smoke seed-{seed} summary for {condition}"
    )


def _comparison_figure(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['arm']}\nγ={row['gamma']:g}" for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(12.0, 9.5), sharex=True)
    metrics = (
        ("reward_percentage", "state-2 occupancy (%)", (0.0, 100.0)),
        ("r2_global", "global transducer belief R²", (0.0, 1.0)),
        ("r2_fine", "fine transducer belief R²", (0.0, 1.0)),
    )
    colors = ["#4C78A8" if row["gamma"] == 0.0 else "#F58518" for row in rows]
    for axis, (key, ylabel, ylim) in zip(axes, metrics):
        axis.bar(x, [row[key] for row in rows], color=colors)
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, labels)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _checkpoint_figure(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 10.0), sharex=True)
    metrics = (
        ("reward_percentage", "state-2 occupancy (%)", (0.0, 100.0)),
        ("r2_global", "global transducer belief R²", (0.0, 1.0)),
        ("r2_fine", "fine transducer belief R²", (0.0, 1.0)),
    )
    for row in rows:
        checkpoints = row["checkpoint_probes"]
        steps = [point["agent_steps"] / 1_000_000 for point in checkpoints]
        label = f"{row['arm']}, γ={row['gamma']:g}"
        for axis, (key, ylabel, ylim) in zip(axes, metrics):
            axis.plot(
                steps,
                [point[key] for point in checkpoints],
                marker="o",
                label=label,
            )
            axis.set_ylabel(ylabel)
            axis.set_ylim(*ylim)
            axis.grid(alpha=0.2)
    axes[-1].set_xlabel("environment steps (millions)")
    axes[0].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("study synthesis requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    family = Path(__file__).parents[1]
    rows = []
    sources = {}
    for condition, arm, gamma in CONDITIONS:
        source, summary = _latest_summary(family, condition, context.seed)
        if float(summary["gamma"]) != gamma:
            raise AssertionError(f"{condition} reported an unexpected gamma")
        rows.append(
            {
                "condition": condition,
                "arm": arm,
                "gamma": gamma,
                "reward_percentage": float(summary["reward_percentage"]),
                "greedy_reward_percentage": float(
                    summary["greedy_reward_percentage"]
                ),
                "r2_global": float(summary["r2_global"]),
                "r2_fine": float(summary["r2_fine"]),
                "checkpoint_probes": [
                    {
                        "agent_steps": int(point["agent_steps"]),
                        "reward_percentage": float(point["reward_percentage"]),
                        "greedy_reward_percentage": float(
                            point["greedy_reward_percentage"]
                        ),
                        "r2_global": float(point["r2_global"]),
                        "r2_fine": float(point["r2_fine"]),
                    }
                    for point in summary["checkpoint_probes"]
                ],
            }
        )
        sources[condition] = str(source.relative_to(family))

    comparison = {
        "seed": context.seed,
        "probe_target": "predictive_transducer_belief",
        "reward_metric": "percentage of held-out policy steps in state 2",
        "conditions": rows,
        "sources": sources,
    }
    outputs.write_json("comparison_summary.json", comparison)
    _comparison_figure(rows, context.results_dir / "comparison.png")
    _checkpoint_figure(
        rows,
        context.results_dir / "checkpoint_comparison.png",
    )
    lines = [
        "# MESS3 reward-state Kelly/IQN battery",
        "",
        (
            "All conditions use 30 million environment steps and seed "
            f"{context.seed}. Belief R² uses the action-aware predictive "
            "transducer target."
        ),
        "",
        "| arm | gamma | reward % | greedy reward % | global R² | fine R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['gamma']:g} | "
            f"{row['reward_percentage']:.2f} | "
            f"{row['greedy_reward_percentage']:.2f} | "
            f"{row['r2_global']:.4f} | {row['r2_fine']:.4f} |"
        )
    lines.append("")
    (context.results_dir / "findings.md").write_text("\n".join(lines))
    return comparison
