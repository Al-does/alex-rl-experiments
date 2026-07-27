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
    ALL_POINTS,
    point_by_name,
)
from experiments.mess3_token_guess_cycle_1.process_design import evaluate
from harness.artifacts import RunArtifacts
from harness.context import RunContext


TRAINED_ARMS = ("ppo", "iqn")


def _validation_results_roots() -> tuple[Path, ...]:
    family = Path(__file__).parents[1]
    return (
        family / "operating_point_validation" / "results",
        family / "fractal_preserving_validation" / "results",
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
    for point in ALL_POINTS:
        if not any(name == point.name for name, _ in grouped):
            continue
        untrained = grouped.get((point.name, "untrained"))
        measured = (
            float(np.mean([probe["r_squared"] for probe in untrained]))
            if untrained
            else None
        )
        reference = probe_floor(point.name, measured)
        if untrained:
            reference["untrained_within_branch_r2"] = float(
                np.mean(
                    [
                        probe["r_squared_within_branch_depth2"]
                        for probe in untrained
                    ]
                )
            )
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
            within = np.array(
                [probe["r_squared_within_branch_depth2"] for probe in probes]
            )
            arms[arm] = {
                "seeds": len(probes),
                "r_squared_mean": float(r_squared.mean()),
                "r_squared_std": float(r_squared.std(ddof=0)),
                "within_branch_r2_mean": float(within.mean()),
                "within_branch_r2_std": float(within.std(ddof=0)),
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


ARM_COLOURS = {"ppo": "#b4413c", "iqn": "#2a7f5f"}


def _grouped_bars(axis, names, summary, mean_key, std_key, width=0.34):
    positions = np.arange(len(names))
    for index, (arm, colour) in enumerate(ARM_COLOURS.items()):
        offset = (index - 0.5) * width
        values, errors = [], []
        for name in names:
            arms = summary["operating_points"][name]["arms"]
            values.append(arms.get(arm, {}).get(mean_key, np.nan))
            errors.append(arms.get(arm, {}).get(std_key, 0.0))
        axis.bar(
            positions + offset, values, width, yerr=errors, capsize=4,
            label=arm.upper(), color=colour, zorder=3,
        )
    axis.grid(axis="y", alpha=0.2)
    return positions


def _plot(summary: dict[str, Any], *, path: Path) -> None:
    names = list(summary["operating_points"])
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    axis = axes[0]
    positions = _grouped_bars(
        axis, names, summary, "r_squared_mean", "r_squared_std"
    )
    for index, name in enumerate(names):
        entry = summary["operating_points"][name]
        axis.add_patch(
            plt.Rectangle(
                (index - 0.5, entry["floor"]), 1.0, 1.0 - entry["floor"],
                color="#2a7f5f", alpha=0.13, zorder=0,
            )
        )
        for key, style in (
            ("floor_window8", ":"),
            ("floor_argmax_cell", "--"),
            ("floor_untrained_network", "-"),
        ):
            if key in entry:
                axis.plot(
                    [index - 0.5, index + 0.5], [entry[key]] * 2,
                    color="black", lw=1.6, ls=style, zorder=4,
                )
        axis.text(
            index + 0.47, 1.0 - (1.0 - entry["floor"]) / 2.0,
            f"band\n{entry['band']:.3f}",
            ha="right", va="center", fontsize=9, style="italic", zorder=5,
        )
    axis.set_xticks(
        positions,
        [
            f"{name}\n{summary['operating_points'][name]['stay']}"
            f" / {summary['operating_points'][name]['alpha']}"
            for name in names
        ],
        fontsize=8.5,
    )
    axis.set_ylim(0.6, 1.03)
    axis.set_ylabel("held-out belief-probe R²")
    axis.set_title(
        "Belief probe\nshaded = band above every no-learning floor", fontsize=11
    )
    axis.legend(loc="lower left")

    axis = axes[1]
    positions = _grouped_bars(
        axis, names, summary, "within_branch_r2_mean", "within_branch_r2_std"
    )
    for index, name in enumerate(names):
        entry = summary["operating_points"][name]
        if "untrained_within_branch_r2" in entry:
            axis.plot(
                [index - 0.5, index + 0.5],
                [entry["untrained_within_branch_r2"]] * 2,
                color="black", lw=1.8, zorder=4,
            )
    axis.axhline(0.0, color="black", lw=1.0, alpha=0.5)
    axis.set_ylim(-1.6, 1.15)
    axis.set_xticks(positions, names, fontsize=8.5)
    axis.set_ylabel("R² within last-two-token branches")
    axis.set_title(
        "Belief detail the token window cannot supply\nblack = untrained network",
        fontsize=11,
    )
    axis.legend(loc="lower left")

    axis = axes[2]
    positions = _grouped_bars(
        axis,
        names,
        summary,
        "accuracy_fraction_of_range_mean",
        "accuracy_fraction_of_range_std",
    )
    axis.axhline(0.0, color="black", lw=1.5)
    axis.axhline(1.0, color="#2a7f5f", lw=1.5, ls="--")
    axis.set_xticks(
        positions,
        [
            f"{name}\nrange"
            f" {summary['operating_points'][name]['accuracy_range']:.4f}"
            for name in names
        ],
        fontsize=8.5,
    )
    axis.set_ylabel("share of the echo-to-Bayes accuracy range")
    axis.set_title(
        "Accuracy, normalised by the range that exists\n"
        "0 = echo the last token, 1 = Bayes optimal",
        fontsize=11,
    )
    axis.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(path, dpi=190)
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
                "usable band, and by "
                f"{entry['iqn_minus_ppo_accuracy_share'] * 100:.0f} points of "
                "the accuracy range."
            )
    lines.extend(
        [
            "",
            "A gap wider than the band is not a larger effect, it is an "
            "off-scale reading: the arms differ by more than the range in "
            "which a difference is interpretable. Seed spread is not "
            "comparable across points either, because an arm pinned at a "
            "non-learning fixed point has almost no variance while an arm that "
            "sometimes learns has a lot.",
            "",
        ]
    )
    return "\n".join(lines)


def run(context: RunContext):
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    cells = load_cells(*_validation_results_roots())
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
