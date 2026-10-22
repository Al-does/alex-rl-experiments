"""Re-analyse the committed multi-seed token-guess results.

This reads the results already in the repository and re-reports them against the
task's own floors and ceilings, with intervals that account for the three seeds
they were measured on. It trains nothing. Its purpose is to establish which of
the published orderings the existing evidence actually supports, so that cycle 2
can be scoped to the questions that remain open.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_token_guess_cycle_2.statistics import (
    compare,
    holm_adjust,
    summarise,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext

STUDIES = ("mess_3_kelly_cycle_2", "mess_3_kelly_cycle_3")
METRICS = ("r_squared", "token_accuracy_greedy")
FALLBACK_REFERENCES = {
    "belief_r2_floor": 0.9668,
    "belief_r2_ceiling": 0.99888,
    "accuracy_floor": 0.6732,
    "accuracy_ceiling": 0.6883,
    "untrained_module_r2": 0.8733,
}


def _references(experiment_dir: Path) -> dict[str, float]:
    """Load the most recent reference run, or fall back to recorded values."""

    candidates = sorted(
        (experiment_dir.parent / "references" / "results").glob("*/references.json")
    )
    if not candidates:
        return dict(FALLBACK_REFERENCES)
    data = json.loads(candidates[-1].read_text())
    untrained = data.get("untrained_module") or {}
    return {
        "belief_r2_floor": float(data["belief_r2_floor"]),
        "belief_r2_ceiling": float(data["supervised_ceiling"]),
        "accuracy_floor": float(data["accuracy_floor_repeat_previous_token"]),
        "accuracy_ceiling": float(data["accuracy_ceiling_bayes"]),
        "untrained_module_r2": float(
            untrained.get("r_squared", FALLBACK_REFERENCES["untrained_module_r2"])
        ),
    }


def collect_study(repository_root: Path, study: str) -> dict[str, dict[int, dict[str, float]]]:
    """Read every ``condition_summary.json`` a study committed, keyed by seed."""

    per_arm: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    pattern = str(repository_root / "experiments" / study / "*" / "results" / "*" / "condition_summary.json")
    for path in sorted(glob.glob(pattern)):
        summary = json.loads(Path(path).read_text())
        probe = summary.get("probe") or {}
        seed = summary.get("seed")
        if seed is None or "r_squared" not in probe:
            continue
        arm = Path(path).parents[2].name
        per_arm[arm][int(seed)] = {
            metric: float(probe[metric]) for metric in METRICS if metric in probe
        }
    return dict(per_arm)


def analyse_study(
    per_arm: dict[str, dict[int, dict[str, float]]],
    references: dict[str, float],
) -> dict[str, Any]:
    """Summarise each arm and test every pairwise ordering within the study."""

    floor = references["belief_r2_floor"]
    ceiling = references["belief_r2_ceiling"]
    conditions: dict[str, Any] = {}
    r2_by_arm: dict[str, dict[int, float]] = {}
    for arm, by_seed in sorted(per_arm.items()):
        values = [by_seed[seed]["r_squared"] for seed in sorted(by_seed)]
        if len(values) < 2:
            continue
        estimate = summarise(values)
        r2_by_arm[arm] = {seed: by_seed[seed]["r_squared"] for seed in by_seed}
        conditions[arm] = {
            "seeds": sorted(by_seed),
            "r_squared_mean": estimate.mean,
            "r_squared_sample_sd": estimate.sample_sd,
            "r_squared_ci": [estimate.ci_low, estimate.ci_high],
            "fraction_of_usable_range": (estimate.mean - floor) / (ceiling - floor),
            "exceeds_no_network_floor": bool(estimate.ci_low > floor),
        }

    comparisons: dict[str, Any] = {}
    arms = sorted(r2_by_arm)
    raw_p: dict[str, float] = {}
    for index, left in enumerate(arms):
        for right in arms[index + 1:]:
            name = f"{left} vs {right}"
            result = compare(r2_by_arm[left], r2_by_arm[right])
            raw_p[name] = result.p_value
            comparisons[name] = {
                "difference": result.difference,
                "difference_ci": [result.ci_low, result.ci_high],
                "paired_sample_sd": result.sample_sd,
                "n_shared_seeds": result.n,
                "p_value": result.p_value,
                "seeds_for_80_percent_power": result.seeds_for_power,
            }
    for name, adjusted in holm_adjust(raw_p).items():
        comparisons[name]["p_value_holm"] = adjusted
        comparisons[name]["resolved"] = bool(
            adjusted < 0.05 and comparisons[name]["difference_ci"][0] > 0.0
        ) or bool(adjusted < 0.05 and comparisons[name]["difference_ci"][1] < 0.0)
    return {"conditions": conditions, "comparisons": comparisons}


def plot_against_references(
    analysis: dict[str, Any],
    references: dict[str, float],
    *,
    untrained_r2: float,
    path: Path,
) -> None:
    """Place every arm's interval inside the range the metric can move through."""

    floor = references["belief_r2_floor"]
    ceiling = references["belief_r2_ceiling"]
    rows: list[tuple[str, float, float, float]] = []
    for study, result in analysis.items():
        label = study.replace("mess_3_", "").replace("_", " ")
        for arm, values in sorted(
            result["conditions"].items(),
            key=lambda item: item[1]["r_squared_mean"],
        ):
            low, high = values["r_squared_ci"]
            rows.append(
                (f"{arm.replace('_', ' ')}\n({label})", values["r_squared_mean"], low, high)
            )

    figure, axis = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 2.6))
    positions = range(len(rows))
    axis.axvspan(floor, ceiling, color="tab:green", alpha=0.10)
    axis.axvline(
        floor,
        color="tab:green",
        linestyle="--",
        linewidth=1.4,
        label=f"affine probe on raw observations ({floor:.4f})",
    )
    axis.axvline(
        untrained_r2,
        color="tab:orange",
        linestyle=":",
        linewidth=1.4,
        label=f"randomly initialised transformer ({untrained_r2:.4f})",
    )
    axis.axvline(
        ceiling,
        color="tab:blue",
        linestyle="-.",
        linewidth=1.4,
        label=f"supervised next-token model ({ceiling:.4f})",
    )
    for position, (_, mean, low, high) in zip(positions, rows):
        cleared = low > floor
        axis.plot(
            [low, high],
            [position, position],
            color="tab:green" if cleared else "tab:red",
            linewidth=2.0,
            alpha=0.75,
        )
        axis.plot(
            mean,
            position,
            "o",
            color="tab:green" if cleared else "tab:red",
            markersize=6,
        )
    axis.set_yticks(list(positions), [row[0] for row in rows], fontsize=8)
    axis.set_xlim(0.72, 1.02)
    axis.set_xlabel("held-out belief-probe R² (95% CI over three seeds)")
    axis.set_title(
        "Committed token-guess results against the range the metric can move through"
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower left", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _findings(analysis: dict[str, Any], references: dict[str, float]) -> str:
    floor = references["belief_r2_floor"]
    ceiling = references["belief_r2_ceiling"]
    lines = [
        "# Audit of the committed multi-seed token-guess results",
        "",
        "Belief-probe R² is reported against the range it can move through: 0% is "
        f"an affine probe on the raw observations ({floor:.4f}) and 100% is the "
        f"supervised next-token replication ({ceiling:.4f}).",
        "",
    ]
    for study, result in analysis.items():
        lines.extend(
            [
                f"## `{study}`",
                "",
                "| condition | R² | 95% CI | usable range | above floor |",
                "|---|---:|---|---:|---|",
            ]
        )
        ordered = sorted(
            result["conditions"].items(),
            key=lambda item: -item[1]["r_squared_mean"],
        )
        for arm, values in ordered:
            low, high = values["r_squared_ci"]
            above = "yes" if values["exceeds_no_network_floor"] else "no"
            lines.append(
                f"| `{arm}` | {values['r_squared_mean']:.4f} | "
                f"[{low:.4f}, {high:.4f}] | "
                f"{values['fraction_of_usable_range']:+.0%} | {above} |"
            )
        resolved = [
            name
            for name, values in result["comparisons"].items()
            if values.get("resolved")
        ]
        total = len(result["comparisons"])
        lines.extend(
            [
                "",
                f"Of {total} pairwise orderings, {len(resolved)} survive a "
                "Holm correction across the family.",
                "",
                "| comparison | difference | 95% CI | Holm p | seeds for 80% power |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for name, values in sorted(
            result["comparisons"].items(), key=lambda item: item[1]["p_value"]
        ):
            low, high = values["difference_ci"]
            needed = values["seeds_for_80_percent_power"]
            lines.append(
                f"| {name} | {values['difference']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | "
                f"{values['p_value_holm']:.3f} | "
                f"{needed if needed < 10_000 else '>10000'} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(context: RunContext) -> dict[str, Any]:
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    repository_root = Path(__file__).parents[3]
    references = _references(Path(__file__).parents[1])
    analysis = {
        study: analyse_study(
            collect_study(repository_root, study),
            references,
        )
        for study in STUDIES
    }
    analysis = {study: result for study, result in analysis.items() if result["conditions"]}
    if not analysis:
        raise RuntimeError("no committed multi-seed results were found to audit")
    figure_path = context.results_dir / "results_against_references.png"
    plot_against_references(
        analysis,
        references,
        untrained_r2=references["untrained_module_r2"],
        path=figure_path,
    )
    result = {
        "references": references,
        "studies": analysis,
        "figure": str(figure_path),
    }
    outputs.write_json("audit.json", result)
    (context.results_dir / "findings.md").write_text(_findings(analysis, references))
    return result
