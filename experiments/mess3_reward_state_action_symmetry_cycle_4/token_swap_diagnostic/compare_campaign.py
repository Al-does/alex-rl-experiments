"""Compare token-swap diagnostics with existing real-target probe summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

SEEDS = (42, 43, 44, 45, 46)
VARIANT = 2


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _token_swap_path(root: Path, cycle: int, seed: int) -> Path:
    return (
        root
        / "token_swap_diagnostic"
        / "results"
        / f"mess3-rsa-c{cycle}-token-swap-v{VARIANT}-seed{seed}"
        / "token_swap_diagnostic.json"
    )


def _full_probe_path(root: Path, cycle: int, seed: int) -> Path:
    return (
        root
        / f"variant_{VARIANT}"
        / "results"
        / f"mess3-rsa-c{cycle}-v{VARIANT}-seed{seed}"
        / "condition_summary.json"
    )


def _symmetry_probe_path(root: Path, cycle: int, seed: int) -> Path | None:
    if cycle != 4:
        return None
    suffix = "0035"
    candidates = (
        root
        / "belief_symmetry_probes"
        / f"variant_{VARIANT}"
        / "results"
        / f"mess3-rsa-c{cycle}-belief-symmetry-probe-{suffix}-v{VARIANT}-seed{seed}"
        / "condition_summary.json",
        root
        / "belief_symmetry_probes"
        / f"variant_{VARIANT}"
        / "results"
        / f"mess3-rsa-c{cycle}-belief-symmetry-probe-v{VARIANT}-seed{seed}"
        / "condition_summary.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def aggregate(root: Path, *, cycle: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        token_swap_path = _token_swap_path(root, cycle, seed)
        full_probe_path = _full_probe_path(root, cycle, seed)
        symmetry_probe_path = _symmetry_probe_path(root, cycle, seed)
        if not token_swap_path.is_file():
            raise FileNotFoundError(f"missing token-swap result: {token_swap_path}")
        if not full_probe_path.is_file():
            raise FileNotFoundError(f"missing full-probe summary: {full_probe_path}")

        token_swap = _load_json(token_swap_path)
        full_probe = _load_json(full_probe_path)
        symmetry_probe = (
            _load_json(symmetry_probe_path)
            if symmetry_probe_path is not None and symmetry_probe_path.is_file()
            else None
        )
        metrics = token_swap["metrics"]
        antisymmetric = None
        symmetric = None
        coarse = None
        if symmetry_probe is not None:
            final_targets = symmetry_probe["checkpoints"]["final"]["targets"]
            antisymmetric = final_targets["antisymmetric_b0_minus_b1"]
            symmetric = final_targets["symmetric_b2"]
            coarse = final_targets.get("coarse_b2")

        rows.append(
            {
                "seed": seed,
                "token_swap": {
                    "factual_mse": metrics["factual"]["mse"],
                    "counterfactual_mse": metrics["counterfactual"]["mse"],
                    "counterfactual_minus_factual_mse": metrics[
                        "counterfactual_minus_factual_mse"
                    ],
                    "counterfactual_over_factual_mse": metrics[
                        "counterfactual_over_factual_mse"
                    ],
                    "equivariance_mse": metrics["equivariance_mse"],
                    "antisymmetric_sign_reversal_rmse": metrics[
                        "antisymmetric_sign_reversal_rmse"
                    ],
                    "state_2_invariance_rmse": metrics["state_2_invariance_rmse"],
                    "shift_mse": metrics["shift_mse"],
                    "shift_cosine_mean": metrics["shift_cosine_mean"],
                    "activation_shift_rms": metrics["activation_shift_rms"],
                },
                "full_probe_final": {
                    "mse": full_probe["final_probe"]["mse"],
                    "global_mse_ratio": full_probe["final_probe"]["global_mse_ratio"],
                    "r_squared": full_probe["final_probe"]["r_squared"],
                },
                "symmetry_probe_final": None
                if antisymmetric is None
                else {
                    "antisymmetric_mse": antisymmetric["mse"],
                    "antisymmetric_global_mse_ratio": antisymmetric[
                        "global_mse_ratio"
                    ],
                    "symmetric_mse": symmetric["mse"],
                    "symmetric_global_mse_ratio": symmetric["global_mse_ratio"],
                    "coarse_mse": None if coarse is None else coarse["mse"],
                    "coarse_global_mse_ratio": None
                    if coarse is None
                    else coarse["global_mse_ratio"],
                },
            }
        )

    def collect(key_path: list[str]) -> list[float]:
        values: list[float] = []
        for row in rows:
            current: Any = row
            for key in key_path:
                current = current[key]
            values.append(float(current))
        return values

    summary = {
        "schema_version": 1,
        "cycle": cycle,
        "variant": VARIANT,
        "seeds": list(SEEDS),
        "rows": rows,
        "aggregate": {
            "token_swap_factual_mse": _stats(
                collect(["token_swap", "factual_mse"])
            ),
            "token_swap_counterfactual_minus_factual_mse": _stats(
                collect(["token_swap", "counterfactual_minus_factual_mse"])
            ),
            "token_swap_equivariance_mse": _stats(
                collect(["token_swap", "equivariance_mse"])
            ),
            "token_swap_antisymmetric_sign_reversal_rmse": _stats(
                collect(["token_swap", "antisymmetric_sign_reversal_rmse"])
            ),
            "full_probe_final_mse": _stats(
                collect(["full_probe_final", "mse"])
            ),
        },
    }
    if rows[0]["symmetry_probe_final"] is not None:
        summary["aggregate"]["symmetry_probe_antisymmetric_mse"] = _stats(
            collect(["symmetry_probe_final", "antisymmetric_mse"])
        )
        summary["aggregate"]["symmetry_probe_symmetric_mse"] = _stats(
            collect(["symmetry_probe_final", "symmetric_mse"])
        )
    return summary


def _findings(summary: dict[str, Any]) -> str:
    agg = summary["aggregate"]
    lines = [
        f"# Cycle {summary['cycle']} variant 2 token-swap versus real-target probes",
        "",
        "Token-swap diagnostics replay fixed greedy histories through frozen final "
        "checkpoints with token channels 0 and 1 exchanged. Full-probe and "
        "symmetry-probe rows come from existing published summaries on the same seeds.",
        "",
        "## Aggregate comparison",
        "",
        f"- Token-swap factual MSE: {agg['token_swap_factual_mse']['mean']:.6f} "
        f"± {agg['token_swap_factual_mse']['stdev']:.6f}",
        f"- Token-swap ΔMSE (counterfactual − factual): "
        f"{agg['token_swap_counterfactual_minus_factual_mse']['mean']:.6f} "
        f"± {agg['token_swap_counterfactual_minus_factual_mse']['stdev']:.6f}",
        f"- Token-swap equivariance MSE: "
        f"{agg['token_swap_equivariance_mse']['mean']:.6f} "
        f"± {agg['token_swap_equivariance_mse']['stdev']:.6f}",
        f"- Token-swap antisymmetric sign-reversal RMSE: "
        f"{agg['token_swap_antisymmetric_sign_reversal_rmse']['mean']:.6f} "
        f"± {agg['token_swap_antisymmetric_sign_reversal_rmse']['stdev']:.6f}",
        f"- Full 3D belief probe final MSE: "
        f"{agg['full_probe_final_mse']['mean']:.6f} "
        f"± {agg['full_probe_final_mse']['stdev']:.6f}",
    ]
    if "symmetry_probe_antisymmetric_mse" in agg:
        lines.extend(
            [
                f"- Symmetry-probe antisymmetric target MSE: "
                f"{agg['symmetry_probe_antisymmetric_mse']['mean']:.6f} "
                f"± {agg['symmetry_probe_antisymmetric_mse']['stdev']:.6f}",
                f"- Symmetry-probe symmetric-b2 target MSE: "
                f"{agg['symmetry_probe_symmetric_mse']['mean']:.6f} "
                f"± {agg['symmetry_probe_symmetric_mse']['stdev']:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-seed rows",
            "",
            "| seed | swap factual MSE | swap ΔMSE | equivariance MSE | "
            "sign-reversal RMSE | full-probe MSE | antisym probe MSE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        antisym = row["symmetry_probe_final"]
        antisym_mse = (
            f"{antisym['antisymmetric_mse']:.6f}" if antisym is not None else "n/a"
        )
        lines.append(
            f"| {row['seed']} | "
            f"{row['token_swap']['factual_mse']:.6f} | "
            f"{row['token_swap']['counterfactual_minus_factual_mse']:.6f} | "
            f"{row['token_swap']['equivariance_mse']:.6f} | "
            f"{row['token_swap']['antisymmetric_sign_reversal_rmse']:.6f} | "
            f"{row['full_probe_final']['mse']:.6f} | "
            f"{antisym_mse} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: low equivariance MSE and sign-reversal RMSE indicate "
            "the decoded representation exchanges b0/b1 under the exact token swap. "
            "A large ΔMSE rejects the hypothesis that counterfactual decoding stays "
            "as accurate as factual decoding.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, cycle: int = 4) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "campaign",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(args.root, cycle=cycle)
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output / "findings.md").write_text(_findings(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
