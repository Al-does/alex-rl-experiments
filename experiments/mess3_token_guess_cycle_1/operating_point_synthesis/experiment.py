"""Score both operating points as instruments, using the trained arms.

A wider axis is only worth having if real arms spread across it. This leaf
places every trained cell inside the band its own operating point makes
available, so the two points can be compared on the thing that matters: how
much of a *usable* range separates arms that are known to differ.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from envs.mess3.model import emission_matrix, symmetric_transition_matrix
from experiments.mess3_token_guess_cycle_1.operating_point_validation.experiment import (
    load_cells,
)
from experiments.mess3_token_guess_cycle_1.operating_points import (
    POINTS,
    point_by_name,
)
from experiments.mess3_token_guess_cycle_1.process_design import evaluate
from harness.artifacts import RunArtifacts
from harness.context import RunContext


TRAINED_ARMS = ("ppo", "iqn")


def _validation_results_root() -> Path:
    return (
        Path(__file__).parents[1] / "operating_point_validation" / "results"
    )


def probe_floor(point_name: str, measured_untrained: float | None) -> dict[str, float]:
    """The highest score reachable without representing the belief."""

    point = point_by_name(point_name)
    design = evaluate(
        symmetric_transition_matrix(point.stay),
        emission_matrix(point.alpha),
        name=point.name,
    )
    floors = {
        "window8": design["probe_r2_window8"],
        "argmax_cell": design["probe_r2_argmax_cell"],
    }
    if measured_untrained is not None:
        floors["untrained_network"] = measured_untrained
    binding = max(floors.values())
    return {
        **{f"floor_{key}": value for key, value in floors.items()},
        "floor": binding,
        "band": design["probe_r2_sufficient"] - binding,
        "accuracy_range": design["accuracy_headroom"],
    }


def summarise(cells: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["operating_point"], cell["arm"])].append(cell["probe"])

    points: dict[str, Any] = {}
    for point in POINTS:
        untrained = grouped.get((point.name, "untrained"))
        measured = (
            float(np.mean([probe["r_squared"] for probe in untrained]))
            if untrained
            else None
        )
        reference = probe_floor(point.name, measured)
        arms: dict[str, Any] = {}
        for arm in TRAINED_ARMS:
            probes = grouped.get((point.name, arm))
            if not probes:
                continue
            r_squared = np.array([probe["r_squared"] for probe in probes])
            share = np.array(
                [probe["accuracy_fraction_of_range"] for probe in probes]
            )
            accuracy = np.array(
                [probe["expected_accuracy_agent"] for probe in probes]
            )
            arms[arm] = {
                "seeds": len(probes),
                "r_squared_mean": float(r_squared.mean()),
                "r_squared_std": float(r_squared.std(ddof=0)),
                "probe_position": float(
                    (r_squared.mean() - reference["floor"])
                    / max(reference["band"], 1e-9)
                ),
                "clears_floor": bool(r_squared.min() > reference["floor"]),
                "expected_accuracy_mean": float(accuracy.mean()),
                "accuracy_fraction_of_range_mean": float(share.mean()),
                "accuracy_fraction_of_range_std": float(share.std(ddof=0)),
            }
        entry: dict[str, Any] = {
            "stay": point.stay,
            "alpha": point.alpha,
            **reference,
            "untrained_r_squared": measured,
            "arms": arms,
        }
        if all(arm in arms for arm in TRAINED_ARMS):
            gap = (
                arms["iqn"]["r_squared_mean"] - arms["ppo"]["r_squared_mean"]
            )
            spread = max(
                arms["iqn"]["r_squared_std"], arms["ppo"]["r_squared_std"], 1e-9
            )
            entry["iqn_minus_ppo_r_squared"] = float(gap)
            entry["iqn_minus_ppo_in_seed_sigma"] = float(gap / spread)
            entry["iqn_minus_ppo_share_of_band"] = float(
                gap / max(reference["band"], 1e-9)
            )
            entry["iqn_minus_ppo_accuracy_share"] = float(
                arms["iqn"]["accuracy_fraction_of_range_mean"]
                - arms["ppo"]["accuracy_fraction_of_range_mean"]
            )
        points[point.name] = entry
    return {"operating_points": points}


def _plot(summary: dict[str, Any], *, path: Path) -> None:
    names = [name for name in summary["operating_points"]]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    axis = axes[0]
    positions = np.arange(len(names))
    width = 0.34
    for offset, arm, colour in (
        (-width / 2, "ppo", "#b4413c"),
        (width / 2, "iqn", "#2a7f5f"),
    ):
        values, errors = [], []
        for name in names:
            arms = summary["operating_points"][name]["arms"]
            values.append(arms.get(arm, {}).get("r_squared_mean", np.nan))
            errors.append(arms.get(arm, {}).get("r_squared_std", 0.0))
        axis.bar(
            positions + offset, values, width, yerr=errors, capsize=4,
            label=arm.upper(), color=colour,
        )
    for index, name in enumerate(names):
        entry = summary["operating_points"][name]
        axis.plot(
            [index - 0.5, index + 0.5], [entry["floor"]] * 2,
            color="black", lw=2.0,
        )
        axis.text(
            index, entry["floor"] + 0.004,
            f"floor {entry['floor']:.3f}",
            ha="center", fontsize=9,
        )
    axis.set_xticks(positions, [
        f"{name}\nstay={summary['operating_points'][name]['stay']},"
        f" alpha={summary['operating_points'][name]['alpha']}"
        for name in names
    ])
    axis.set_ylim(0.6, 1.02)
    axis.set_ylabel("held-out belief-probe R²")
    axis.set_title("Belief probe against the no-learning floor")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()

    axis = axes[1]
    for offset, arm, colour in (
        (-width / 2, "ppo", "#b4413c"),
        (width / 2, "iqn", "#2a7f5f"),
    ):
        values, errors = [], []
        for name in names:
            arms = summary["operating_points"][name]["arms"]
            values.append(
                arms.get(arm, {}).get("accuracy_fraction_of_range_mean", np.nan)
            )
            errors.append(
                arms.get(arm, {}).get("accuracy_fraction_of_range_std", 0.0)
            )
        axis.bar(
            positions + offset, values, width, yerr=errors, capsize=4,
            label=arm.upper(), color=colour,
        )
    axis.axhline(0.0, color="black", lw=1.5)
    axis.axhline(1.0, color="#2a7f5f", lw=1.5, ls="--")
    axis.set_xticks(positions, [
        f"{name}\nrange={summary['operating_points'][name]['accuracy_range']:.4f}"
        for name in names
    ])
    axis.set_ylabel("share of the echo-to-Bayes accuracy range")
    axis.set_title("Accuracy, normalised by the range that exists")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _findings(summary: dict[str, Any]) -> str:
    lines = [
        "# Operating-point validation",
        "",
        "Matched PPO and IQN arms, identical recipe, architecture, budget, "
        "seeds, and probe. Only the process differs.",
        "",
        "| point | stay | alpha | floor | band | arm | belief R² | clears floor "
        "| share of accuracy range |",
        "|---|---:|---:|---:|---:|---|---:|:--:|---:|",
    ]
    for name, entry in summary["operating_points"].items():
        for arm, values in entry["arms"].items():
            lines.append(
                f"| {name} | {entry['stay']} | {entry['alpha']} |"
                f" {entry['floor']:.3f} | {entry['band']:.3f} | {arm} |"
                f" {values['r_squared_mean']:.4f} ±"
                f" {values['r_squared_std']:.4f} |"
                f" {'yes' if values['clears_floor'] else 'no'} |"
                f" {values['accuracy_fraction_of_range_mean'] * 100:.0f}% |"
            )
    lines.append("")
    for name, entry in summary["operating_points"].items():
        if "iqn_minus_ppo_r_squared" in entry:
            lines.append(
                f"- `{name}`: IQN beats PPO by "
                f"{entry['iqn_minus_ppo_r_squared']:.4f} belief R², which is "
                f"{entry['iqn_minus_ppo_share_of_band'] * 100:.0f}% of the "
                f"usable band and "
                f"{entry['iqn_minus_ppo_in_seed_sigma']:.1f} seed sigma, and by "
                f"{entry['iqn_minus_ppo_accuracy_share'] * 100:.0f} points of "
                "the accuracy range."
            )
    lines.append("")
    return "\n".join(lines)


def run(context: RunContext):
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    cells = load_cells(_validation_results_root())
    if not cells:
        raise FileNotFoundError(
            "no operating-point validation cells found; run the validation first"
        )
    summary = summarise(cells)
    summary["n_cells"] = len(cells)
    figure_path = context.results_dir / "operating_point_comparison.png"
    _plot(summary, path=figure_path)
    summary["figure"] = str(figure_path)
    outputs.write_json("operating_point_summary.json", summary)
    (context.results_dir / "findings.md").write_text(_findings(summary))
    print(json.dumps(summary["operating_points"], indent=2, default=float))
    return summary
