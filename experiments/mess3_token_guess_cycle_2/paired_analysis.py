"""Aggregate 0.66M truncated runs and run seed-matched paired t-tests.

Usual analyses (comparison table, MSE-over-training bars) plus focused paired
tests:
  1) decoupled_kelly vs predictive_loss
  2) decoupled_kelly vs ppo

Example:

  uv run python -m experiments.mess3_token_guess_cycle_2.paired_analysis \\
    --results-root experiments/mess3_token_guess_cycle_2 \\
    --output-dir experiments/mess3_token_guess_cycle_2/results/CAMPAIGN \\
    --seeds $(seq 42 56) --run-suffix -0p66m \\
    --conditions ppo predictive_loss decoupled_kelly
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from experiments.mess3_token_guess_cycle_2.comparison import write_comparison
from experiments.mess3_token_guess_cycle_2.mse_charts import (
    load_mse_curves,
    write_mse_bar_charts,
)
from experiments.mess3_token_guess_cycle_2.shared import CONDITIONS

FOCUS_PAIRS = (
    ("decoupled_kelly", "predictive_loss"),
    ("decoupled_kelly", "ppo"),
)
DEFAULT_ARMS = ("ppo", "predictive_loss", "decoupled_kelly")
DEFAULT_SEEDS = tuple(range(42, 57))
THIRD_CHECKPOINT_INDEX = 2  # init=0, ~0.33M=1, ~0.66M=2


def _curve_path(
    results_root: Path,
    condition: str,
    seed: int,
    *,
    run_suffix: str,
) -> Path:
    run_name = f"mess3_token_guess_cycle_2-{condition}-seed{seed}{run_suffix}"
    return (
        results_root
        / condition
        / "results"
        / run_name
        / "checkpoint_probe_curve.json"
    )


def _summary_path(
    results_root: Path,
    condition: str,
    seed: int,
    *,
    run_suffix: str,
) -> Path:
    run_name = f"mess3_token_guess_cycle_2-{condition}-seed{seed}{run_suffix}"
    return (
        results_root
        / condition
        / "results"
        / run_name
        / "condition_summary.json"
    )


def load_condition_summaries(
    results_root: Path,
    *,
    conditions: Sequence[str],
    seeds: Sequence[int],
    run_suffix: str,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        runs: list[dict[str, Any]] = []
        for seed in seeds:
            path = _summary_path(
                results_root, condition, seed, run_suffix=run_suffix
            )
            payload = json.loads(path.read_text())
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path} is not a JSON object")
            runs.append(dict(payload))
        out[condition] = runs
    return out


def _metric_at_checkpoint(
    curve_points: Sequence[Mapping[str, Any]],
    *,
    index: int,
    key: str,
) -> float:
    if index < 0 or index >= len(curve_points):
        raise ValueError(
            f"checkpoint index {index} out of range for {len(curve_points)} points"
        )
    point = curve_points[index]
    if key in point:
        return float(point[key])
    probe = point.get("probe")
    if isinstance(probe, Mapping) and key in probe:
        return float(probe[key])
    raise KeyError(key)


def paired_ttest(
    *,
    candidate: str,
    control: str,
    candidate_values: Sequence[float],
    control_values: Sequence[float],
    seeds: Sequence[int],
    metric: str,
    checkpoint_index: int,
    agent_steps: int | None,
) -> dict[str, Any]:
    cand = np.asarray(candidate_values, dtype=float)
    ctrl = np.asarray(control_values, dtype=float)
    if cand.shape != ctrl.shape or cand.size != len(seeds):
        raise ValueError("paired arrays must align with seeds")
    diffs = cand - ctrl
    shapiro = stats.shapiro(diffs) if diffs.size >= 3 else None
    ttest = stats.ttest_rel(cand, ctrl)
    normality_ok = (
        shapiro is not None and float(shapiro.pvalue) >= 0.05
    )
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
    interpretation = (
        f"{candidate} MSE is {'lower' if mean_diff < 0 else 'higher'} than "
        f"{control} on mean paired delta={mean_diff:.6g}; "
        f"paired t={float(ttest.statistic):.6g}, p={float(ttest.pvalue):.6g}."
    )
    per_seed = [
        {
            "seed": int(seed),
            "control": float(ctrl[i]),
            "candidate": float(cand[i]),
            "diff_candidate_minus_control": float(diffs[i]),
        }
        for i, seed in enumerate(seeds)
    ]
    return {
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "checkpoint_index": checkpoint_index,
        "agent_steps": agent_steps,
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "paired_differences_candidate_minus_control": {
            str(int(seed)): float(diffs[i]) for i, seed in enumerate(seeds)
        },
        "mean_paired_difference": mean_diff,
        "sd_paired_difference": sd_diff,
        "shapiro_wilk": (
            {
                "statistic": float(shapiro.statistic),
                "pvalue": float(shapiro.pvalue),
                "alpha": 0.05,
                "normality_passes": bool(normality_ok),
            }
            if shapiro is not None
            else None
        ),
        "paired_t_test": {
            "method": "scipy.stats.ttest_rel(candidate, control)",
            "applicable": bool(normality_ok),
            "reason": (
                "Shapiro–Wilk on paired differences passed at alpha=0.05"
                if normality_ok
                else "Shapiro–Wilk failed or n<3; t-test shown descriptively"
            ),
            "statistic": float(ttest.statistic),
            "pvalue": float(ttest.pvalue),
            "df": int(max(diffs.size - 1, 0)),
        },
        "interpretation": interpretation,
    }


def _write_ttest_markdown(payload: Mapping[str, Any], path: Path) -> None:
    candidate = payload["candidate"]
    control = payload["control"]
    lines = [
        f"# {candidate} vs {control} — paired t-test",
        "",
        f"Control: `{control}`. Candidate: `{candidate}`.",
        "",
        f"Metric: held-out affine probe MSE at checkpoint index "
        f"**{payload['checkpoint_index']}**"
        + (
            f" (~{payload['agent_steps']:,} env steps)."
            if payload.get("agent_steps") is not None
            else "."
        ),
        "",
        f"Design: {len(payload['seeds'])} same-seed pairs "
        f"({payload['seeds'][0]}–{payload['seeds'][-1]}). For each pair, "
        "`diff_i = candidate_i − control_i` (negative ⇒ candidate lower MSE / better).",
        "",
        "## Per-seed values",
        "",
        f"| seed | {control} (control) MSE | {candidate} (candidate) MSE | "
        "diff (candidate − control) |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['control']:.9f} | {row['candidate']:.9f} | "
            f"{row['diff_candidate_minus_control']:+.6e} |"
        )
    lines.extend(
        [
            "",
            f"Mean paired difference: **{payload['mean_paired_difference']:.6e}** "
            f"(SD = {payload['sd_paired_difference']:.6e}).",
            "",
        ]
    )
    shapiro = payload.get("shapiro_wilk")
    if shapiro:
        passed = "passes" if shapiro["normality_passes"] else "fails"
        lines.extend(
            [
                f"## Normality check (Shapiro–Wilk on the {len(payload['seeds'])} differences)",
                "",
                "| statistic | value |",
                "|---|---:|",
                f"| W | {shapiro['statistic']:.6f} |",
                f"| p | {shapiro['pvalue']:.6f} |",
                "",
                f"Normality **{passed}** at α = 0.05.",
                "",
            ]
        )
    ttest = payload["paired_t_test"]
    lines.extend(
        [
            "## Paired t-test",
            "",
            f"`{ttest['method']}`:",
            "",
            "| statistic | value |",
            "|---|---:|",
            f"| t | {ttest['statistic']:.6f} |",
            f"| p | {ttest['pvalue']:.6f} |",
            f"| df | {ttest['df']} |",
            "",
            payload["interpretation"],
            "",
            f"Machine-readable copy: [`{path.with_suffix('.json').name}`]"
            f"({path.with_suffix('.json').name}).",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_focus_paired_tests(
    curves: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    seeds: Sequence[int],
    checkpoint_index: int = THIRD_CHECKPOINT_INDEX,
    pairs: Sequence[tuple[str, str]] = FOCUS_PAIRS,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate, control in pairs:
        cand_vals = [
            _metric_at_checkpoint(
                curves[candidate][seed], index=checkpoint_index, key="mse"
            )
            for seed in seeds
        ]
        ctrl_vals = [
            _metric_at_checkpoint(
                curves[control][seed], index=checkpoint_index, key="mse"
            )
            for seed in seeds
        ]
        agent_steps = int(
            curves[candidate][seeds[0]][checkpoint_index]["agent_steps"]
        )
        payload = paired_ttest(
            candidate=candidate,
            control=control,
            candidate_values=cand_vals,
            control_values=ctrl_vals,
            seeds=seeds,
            metric="held_out_affine_probe_mse",
            checkpoint_index=checkpoint_index,
            agent_steps=agent_steps,
        )
        results.append(payload)
    return results


def _findings(
    *,
    comparison: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    seeds: Sequence[int],
    run_suffix: str,
) -> str:
    lines = [
        "# MESS3 token-guess cycle 2 — 0.66M / 15-seed campaign",
        "",
        f"Truncated runs through the third checkpoint (~0.66M steps), "
        f"seeds {seeds[0]}–{seeds[-1]} (n={len(seeds)}), run suffix "
        f"`{run_suffix or '(none)'}`.",
        "",
        "| condition | seeds | belief R² | token accuracy | mse |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in conditions:
        values = comparison["conditions"][name]
        metrics = values["metrics"]
        lines.append(
            f"| {name} | {len(values['seeds'])} | "
            f"{metrics['r_squared']['mean']:.4f} ± "
            f"{metrics['r_squared']['std']:.4f} | "
            f"{metrics['token_accuracy_greedy']['mean']:.4f} ± "
            f"{metrics['token_accuracy_greedy']['std']:.4f} | "
            f"{metrics['mse']['mean']:.6f} ± "
            f"{metrics['mse']['std']:.6f} |"
        )
    lines.extend(["", "## Focused paired t-tests (same seed)", ""])
    for payload in paired:
        ttest = payload["paired_t_test"]
        lines.append(
            f"- **{payload['candidate']} vs {payload['control']}** at checkpoint "
            f"{payload['checkpoint_index']} (~{payload['agent_steps']:,} steps): "
            f"mean ΔMSE={payload['mean_paired_difference']:.6e}, "
            f"t={ttest['statistic']:.3f}, p={ttest['pvalue']:.4g} "
            f"(df={ttest['df']})."
        )
    lines.append("")
    return "\n".join(lines)


def write_campaign_analyses(
    results_root: Path,
    *,
    output_dir: Path,
    conditions: Sequence[str],
    seeds: Sequence[int],
    run_suffix: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_condition_summaries(
        results_root,
        conditions=conditions,
        seeds=seeds,
        run_suffix=run_suffix,
    )
    # comparison.write_comparison expects all CONDITIONS by default; pass a
    # patched view with only requested arms by writing a local summary.
    comparison_conditions = {
        c.name: c for c in CONDITIONS if c.name in conditions
    }
    # Reuse summarize via a thin local path: write_comparison requires all
    # CONDITIONS keys. Build metrics ourselves for the subset.
    from experiments.mess3_token_guess_cycle_2 import comparison as comparison_mod

    original = comparison_mod.CONDITIONS
    try:
        comparison_mod.CONDITIONS = tuple(
            c for c in original if c.name in conditions
        )
        comparison = write_comparison(summaries, output_dir=output_dir)
    finally:
        comparison_mod.CONDITIONS = original

    curves = load_mse_curves(
        results_root,
        conditions=conditions,
        seeds=seeds,
        run_suffix=run_suffix,
    )
    mse_dir = output_dir / "mse_over_training"
    write_mse_bar_charts(
        curves,
        output_dir=mse_dir,
        conditions=conditions,
        seeds=seeds,
    )
    paired = run_focus_paired_tests(curves, seeds=seeds)
    paired_paths: list[str] = []
    for payload in paired:
        stem = f"{payload['candidate']}_vs_{payload['control']}_paired_ttest"
        json_path = output_dir / f"{stem}.json"
        md_path = output_dir / f"{stem}.md"
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        _write_ttest_markdown(payload, md_path)
        paired_paths.extend([json_path.name, md_path.name])

    findings = _findings(
        comparison=comparison,
        paired=paired,
        conditions=conditions,
        seeds=seeds,
        run_suffix=run_suffix,
    )
    (output_dir / "findings.md").write_text(findings)
    meta = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "results_root": str(results_root),
        "conditions": list(conditions),
        "seeds": list(seeds),
        "run_suffix": run_suffix,
        "paired_outputs": paired_paths,
        "checkpoint_index": THIRD_CHECKPOINT_INDEX,
    }
    (output_dir / "campaign_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Usual cycle-2 analyses plus focused paired t-tests."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path ending at experiments/mess3_token_guess_cycle_2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_ARMS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--run-suffix",
        default="-0p66m",
        help="Suffix on per-seed run folders (default: -0p66m)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_campaign_analyses(
        args.results_root,
        output_dir=args.output_dir,
        conditions=args.conditions,
        seeds=args.seeds,
        run_suffix=args.run_suffix,
    )
    print(f"wrote campaign analyses under {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
