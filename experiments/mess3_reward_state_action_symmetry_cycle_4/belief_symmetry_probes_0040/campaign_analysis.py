"""Aggregate the 3-variant by 3-seed longitudinal belief-probe campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes_0040.analysis import (
    CHECKPOINT_LABELS,
)

SEEDS = (42, 43, 44)
VARIANTS = (1, 2, 3)
TARGETS = ("symmetric_b2", "antisymmetric_b0_minus_b1", "coarse_b2")
CAMPAIGN_SUFFIX = "0040"


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _paired(values: list[float], *, seed: int) -> dict[str, Any]:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(10_000)]
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        **_stats(values),
        "bootstrap_ci_95": [float(low), float(high)],
        "interpretation": "paired descriptive delta; three seeds are insufficient for a strong mechanistic claim",
    }


def _run_path(root: Path, cycle: int, variant: int, seed: int) -> Path:
    run_id = (
        f"mess3-rsa-c{cycle}-belief-symmetry-probe-{CAMPAIGN_SUFFIX}-"
        f"v{variant}-seed{seed}"
    )
    return root / f"variant_{variant}" / "results" / run_id / "condition_summary.json"


def aggregate(root: Path, *, cycle: int) -> dict[str, Any]:
    runs: dict[tuple[int, int], dict[str, Any]] = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = _run_path(root, cycle, variant, seed)
            if not path.is_file():
                raise FileNotFoundError(f"required campaign run missing: {path}")
            runs[(variant, seed)] = json.loads(path.read_text())

    aggregate_targets: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_targets = ("symmetric_b2", "antisymmetric_b0_minus_b1")
        if variant in (1, 2):
            variant_targets += ("coarse_b2",)
        aggregate_targets[f"variant_{variant}"] = {}
        for target in variant_targets:
            checkpoints = {}
            for checkpoint in CHECKPOINT_LABELS:
                entries = [
                    runs[(variant, seed)]["checkpoints"][checkpoint]["targets"][target]
                    for seed in SEEDS
                ]
                checkpoints[checkpoint] = {
                    metric: _stats([float(entry[metric]) for entry in entries])
                    for metric in ("global_mse_ratio", "mse", "r_squared")
                }
                checkpoints[checkpoint]["permutation_null_mse_p50"] = _stats(
                    [float(entry["permutation_null_mse_p50"]) for entry in entries]
                )
                checkpoints[checkpoint][
                    "permutation_null_global_mse_ratio_p50"
                ] = _stats(
                    [
                        float(entry["permutation_null_mse_p50"])
                        / float(entry["target_variance"])
                        for entry in entries
                    ]
                )
                checkpoints[checkpoint]["permutation_p_value"] = _stats(
                    [
                        float(entry["permutation_null_p_value_lower_tail"])
                        for entry in entries
                    ]
                )
            aggregate_targets[f"variant_{variant}"][target] = checkpoints

    paired: dict[str, Any] = {}
    for variant in (1, 2):
        normalized, raw = [], []
        for seed in SEEDS:
            final = runs[(variant, seed)]["checkpoints"]["iter_22"]["targets"]
            normalized.append(
                float(final["coarse_b2"]["global_mse_ratio"])
                - float(final["symmetric_b2"]["global_mse_ratio"])
            )
            raw.append(
                float(final["coarse_b2"]["mse"])
                - float(final["symmetric_b2"]["mse"])
            )
        paired[f"variant_{variant}"] = {
            "coarse_minus_symmetric_global_mse_ratio": _paired(
                normalized, seed=cycle * 100 + variant
            ),
            "coarse_minus_symmetric_raw_mse": _paired(
                raw, seed=cycle * 1000 + variant
            ),
        }
    return {
        "schema_version": 1,
        "cycle": cycle,
        "campaign_suffix": CAMPAIGN_SUFFIX,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "checkpoint_labels": list(CHECKPOINT_LABELS),
        "primary_metric": "global_mse_ratio",
        "comparisons": (
            "Each checkpoint is compared with the initial random-network floor "
            "and each target's independently fitted held-out permutation null."
        ),
        "targets": aggregate_targets,
        "paired_coarse_projection_deltas": paired,
    }


def _plot(summary: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), sharey="row")
    colors = {
        "initial": "#9a9a9a",
        "iter_2": "#6a9e6a",
        "iter_8": "#355c9a",
        "iter_22": "#1a3a6a",
        "permutation": "#c45135",
    }
    targets = (
        "symmetric_b2",
        "antisymmetric_b0_minus_b1",
        "coarse_b2",
    )
    for row, target in enumerate(targets):
        variants = VARIANTS if target != "coarse_b2" else (1, 2)
        for col, variant in enumerate(variants):
            axis = axes[row, col]
            x = np.arange(len(CHECKPOINT_LABELS))
            width = 0.15
            offsets = np.linspace(-0.3, 0.3, len(CHECKPOINT_LABELS) + 1)
            for offset, checkpoint in zip(offsets[:-1], CHECKPOINT_LABELS, strict=True):
                stats = summary["targets"][f"variant_{variant}"][target][checkpoint][
                    "global_mse_ratio"
                ]
                axis.bar(
                    x[CHECKPOINT_LABELS.index(checkpoint)] + offset,
                    stats["mean"],
                    width=width,
                    yerr=stats["stdev"],
                    color=colors[checkpoint],
                    label=checkpoint if col == 0 and row == 0 else None,
                    capsize=2,
                )
            perm = summary["targets"][f"variant_{variant}"][target]["iter_22"][
                "permutation_null_global_mse_ratio_p50"
            ]
            axis.axhline(perm["mean"], color=colors["permutation"], linestyle=":", linewidth=1)
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.set_xticks(x, [label.replace("_", " ") for label in CHECKPOINT_LABELS])
            axis.set_title(f"v{variant} {target.replace('_', ' ')}")
            axis.grid(axis="y", alpha=0.2)
        for col in range(len(variants), axes.shape[1]):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("global MSE ratio (lower is better)")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _findings(summary: dict[str, Any]) -> str:
    lines = [
        f"# Cycle {summary['cycle']} belief-symmetry probe campaign {CAMPAIGN_SUFFIX}",
        "",
        "Longitudinal probes at init, training iterations 2, 8, and 22. "
        "Primary comparisons use held-out global MSE ratio.",
        "",
        "The initial checkpoint is a random-network floor. Each target also has "
        "its own held-out permutation null (dotted line at iter 22 in figures).",
        "",
        "## Symmetry decomposition",
        "",
    ]
    for variant in VARIANTS:
        for target in ("symmetric_b2", "antisymmetric_b0_minus_b1"):
            checkpoints = summary["targets"][f"variant_{variant}"][target]
            initial = checkpoints["initial"]["global_mse_ratio"]
            final = checkpoints["iter_22"]["global_mse_ratio"]
            permutation = checkpoints["iter_22"][
                "permutation_null_global_mse_ratio_p50"
            ]
            direction = (
                "below"
                if final["mean"] < initial["mean"]
                else "at or above"
            )
            lines.append(
                f"- Variant {variant}, `{target}`: final normalized MSE "
                f"{final['mean']:.4f} ± {final['stdev']:.4f}, {direction} the "
                f"random-network value {initial['mean']:.4f} ± "
                f"{initial['stdev']:.4f}; permutation-null median ratio "
                f"{permutation['mean']:.4f} ± {permutation['stdev']:.4f}."
            )
    lines.extend(
        [
            "",
            "## Coarse filter versus projected full filter (variants 1 and 2, iter 22)",
            "",
        ]
    )
    for variant in (1, 2):
        deltas = summary["paired_coarse_projection_deltas"][f"variant_{variant}"]
        normalized = deltas["coarse_minus_symmetric_global_mse_ratio"]
        favored = (
            "coarse filter"
            if normalized["mean"] < 0.0
            else "projected full filter"
        )
        low, high = normalized["bootstrap_ci_95"]
        lines.append(
            f"- Variant {variant}: normalized delta {normalized['mean']:.4f} "
            f"(bootstrap 95% CI [{low:.4f}, {high:.4f}]); favors the {favored}."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, cycle: int = 4) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.root / "results" / "campaign"
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(args.root, cycle=cycle)
    (output / "campaign_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "findings.md").write_text(_findings(summary))
    _plot(summary, output / "belief_symmetry_longitudinal.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
