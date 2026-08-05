"""Aggregate the exact 3-variant by 5-seed belief-probe campaign."""

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

SEEDS = (42, 43, 44, 45, 46)
VARIANTS = (1, 2, 3)
TARGETS = ("symmetric_b2", "antisymmetric_b0_minus_b1", "coarse_b2")
CAMPAIGN_SUFFIX = "0035"


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
        "interpretation": "paired descriptive delta; five seeds are insufficient for a strong mechanistic claim",
    }


def _run_paths(root: Path, cycle: int, variant: int, seed: int) -> tuple[Path, ...]:
    current_run_id = (
        f"mess3-rsa-c{cycle}-belief-symmetry-probe-{CAMPAIGN_SUFFIX}-"
        f"v{variant}-seed{seed}"
    )
    legacy_run_id = f"mess3-rsa-c{cycle}-belief-symmetry-probe-v{variant}-seed{seed}"
    results = root / f"variant_{variant}" / "results"
    return tuple(
        results / run_id / "condition_summary.json"
        for run_id in (current_run_id, legacy_run_id)
    )


def aggregate(root: Path, *, cycle: int) -> dict[str, Any]:
    runs: dict[tuple[int, int], dict[str, Any]] = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            candidates = _run_paths(root, cycle, variant, seed)
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    "required campaign run missing; checked "
                    + ", ".join(str(candidate) for candidate in candidates)
                )
            runs[(variant, seed)] = json.loads(path.read_text())

    aggregate_targets: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_targets = ("symmetric_b2", "antisymmetric_b0_minus_b1")
        if variant in (1, 2):
            variant_targets += ("coarse_b2",)
        aggregate_targets[f"variant_{variant}"] = {}
        for target in variant_targets:
            checkpoints = {}
            for checkpoint in ("initial", "final"):
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
            final = runs[(variant, seed)]["checkpoints"]["final"]["targets"]
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
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "primary_metric": "global_mse_ratio",
        "comparisons": (
            "Final checkpoints are compared with each run's initial random-network "
            "floor and each target's independently fitted held-out permutation null."
        ),
        "targets": aggregate_targets,
        "paired_coarse_projection_deltas": paired,
    }


def _plot(summary: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    colors = {
        "initial": "#9a9a9a",
        "final": "#355c9a",
        "permutation": "#c45135",
    }
    targets = (
        "symmetric_b2",
        "antisymmetric_b0_minus_b1",
        "coarse_b2",
    )
    for axis, target in zip(axes, targets, strict=True):
        variants = VARIANTS if target != "coarse_b2" else (1, 2)
        x = np.arange(len(variants))
        specifications = (
            (-0.24, "initial", "global_mse_ratio"),
            (0.0, "final", "global_mse_ratio"),
            (
                0.24,
                "permutation",
                "permutation_null_global_mse_ratio_p50",
            ),
        )
        for offset, checkpoint, metric in specifications:
            means, errors = [], []
            source_checkpoint = "final" if checkpoint == "permutation" else checkpoint
            for variant in variants:
                stats = summary["targets"][f"variant_{variant}"][target][
                    source_checkpoint
                ][metric]
                means.append(stats["mean"])
                errors.append(stats["stdev"])
            axis.bar(
                x + offset,
                means,
                width=0.23,
                yerr=errors,
                color=colors[checkpoint],
                label=checkpoint,
                capsize=3,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x, [f"v{variant}" for variant in variants])
        axis.set_title(target.replace("_", " "))
        axis.set_xlabel("variant")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("global MSE ratio (lower is better)")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _findings(summary: dict[str, Any]) -> str:
    lines = [
        f"# Cycle {summary['cycle']} belief-symmetry probe campaign",
        "",
        "Primary comparisons use held-out global MSE ratio. Raw MSE and R² are "
        "retained in `campaign_summary.json`; error bars in the figure are seed SD.",
        "",
        "The initial checkpoint is a random-network floor, not a zero-information "
        "control. Each target also has its own held-out permutation null.",
        "",
        "## Symmetry decomposition",
        "",
    ]
    for variant in VARIANTS:
        for target in ("symmetric_b2", "antisymmetric_b0_minus_b1"):
            checkpoints = summary["targets"][f"variant_{variant}"][target]
            initial = checkpoints["initial"]["global_mse_ratio"]
            final = checkpoints["final"]["global_mse_ratio"]
            permutation = checkpoints["final"][
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
            "## Coarse filter versus projected full filter",
            "",
            "Deltas are `coarse - symmetric`, paired by model seed. Negative "
            "values favor the cheap coarse-filter target; positive values favor "
            "the symmetric projection of the full filter. With five seeds these "
            "comparisons are descriptive.",
            "",
        ]
    )
    for variant in (1, 2):
        deltas = summary["paired_coarse_projection_deltas"][f"variant_{variant}"]
        normalized = deltas["coarse_minus_symmetric_global_mse_ratio"]
        raw = deltas["coarse_minus_symmetric_raw_mse"]
        favored = (
            "coarse filter"
            if normalized["mean"] < 0.0
            else "projected full filter"
        )
        low, high = normalized["bootstrap_ci_95"]
        lines.append(
            f"- Variant {variant}: normalized delta {normalized['mean']:.4f} "
            f"(bootstrap 95% CI [{low:.4f}, {high:.4f}]); raw-MSE delta "
            f"{raw['mean']:.6f}. The point estimate favors the {favored}."
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
    _plot(summary, output / "belief_symmetry_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
