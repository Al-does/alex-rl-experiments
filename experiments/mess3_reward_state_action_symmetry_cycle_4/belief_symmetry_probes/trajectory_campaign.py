"""Aggregate and plot one all-checkpoint scalar-probe campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    SEEDS,
    TARGET_VARIANTS,
    TRAJECTORY_SUFFIX,
)


def _run_id(cycle: int, target: str, variant: int, seed: int) -> str:
    return (
        f"mess3-rsa-c{cycle}-belief-trajectory-{TRAJECTORY_SUFFIX}-"
        f"{target.replace('_', '-')}-v{variant}-seed{seed}"
    )


def _load_run(
    root: Path,
    *,
    cycle: int,
    target: str,
    variant: int,
    seed: int,
) -> dict[str, Any]:
    path = (
        root
        / f"variant_{variant}"
        / "results"
        / _run_id(cycle, target, variant, seed)
        / "condition_summary.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"required trajectory run missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("requested_target") != target:
        raise ValueError(f"{path}: expected requested_target={target!r}")
    return payload


def aggregate(root: Path, *, cycle: int, target: str) -> dict[str, Any]:
    if target not in TARGET_VARIANTS:
        raise ValueError(f"unknown trajectory target: {target}")
    variants: dict[str, Any] = {}
    for variant in TARGET_VARIANTS[target]:
        runs = {
            seed: _load_run(
                root,
                cycle=cycle,
                target=target,
                variant=variant,
                seed=seed,
            )
            for seed in SEEDS
        }
        reference_schedule = runs[SEEDS[0]]["checkpoint_schedule"]
        labels = [point["label"] for point in reference_schedule]
        iterations = [point["training_iteration"] for point in reference_schedule]
        seed_curves: dict[str, list[float]] = {}
        for seed, run in runs.items():
            if run["checkpoint_schedule"] != reference_schedule:
                raise ValueError(
                    f"variant {variant} seed {seed} checkpoint schedule differs"
                )
            seed_curves[str(seed)] = [
                float(run["checkpoints"][label]["targets"][target]["global_mse_ratio"])
                for label in labels
            ]
        values = np.asarray(list(seed_curves.values()), dtype=np.float64)
        variants[f"variant_{variant}"] = {
            "checkpoint_labels": labels,
            "training_iterations": iterations,
            "seed_curves": seed_curves,
            "mean": values.mean(axis=0).tolist(),
            "stdev": values.std(axis=0, ddof=1).tolist(),
        }
    return {
        "schema_version": 1,
        "cycle": cycle,
        "target": target,
        "target_definition": {
            "symmetric_b2": "exact full-filter P(state=2), the last belief element",
            "antisymmetric_b0_minus_b1": "exact full-filter P(state=0)-P(state=1)",
            "coarse_b2": (
                "separate two-state HMM filter over A={state0,state1}, B={state2}"
            ),
        }[target],
        "variants": variants,
        "seeds": list(SEEDS),
        "metric": "held-out global MSE / target variance",
        "checkpoint_scope": "initialization and every saved training checkpoint",
    }


def _plot(summary: dict[str, Any], output_stem: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary["variants"])))
    for color, (variant_name, curve) in zip(
        colors, summary["variants"].items(), strict=True
    ):
        x = np.asarray(curve["training_iterations"], dtype=np.float64)
        for seed, values in curve["seed_curves"].items():
            axis.plot(
                x,
                values,
                color=color,
                linewidth=0.9,
                alpha=0.25,
                label=None,
            )
        mean = np.asarray(curve["mean"], dtype=np.float64)
        stdev = np.asarray(curve["stdev"], dtype=np.float64)
        label = variant_name.replace("_", " ")
        axis.plot(x, mean, color=color, linewidth=2.4, marker="o", label=label)
        axis.fill_between(x, mean - stdev, mean + stdev, color=color, alpha=0.13)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="variance baseline")
    axis.set_xlabel("Training iteration (0 = initialization)")
    axis.set_ylabel("Held-out normalized probe MSE (lower is better)")
    axis.set_title(
        f"Cycle {summary['cycle']} — {summary['target'].replace('_', ' ')}"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=200)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def write_campaign(root: Path, *, cycle: int, target: str) -> Path:
    summary = aggregate(root, cycle=cycle, target=target)
    output = root / "results" / "trajectory_campaign" / target
    output.mkdir(parents=True, exist_ok=True)
    (output / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _plot(summary, output / "probe_trajectory")
    return output / "probe_trajectory.png"
