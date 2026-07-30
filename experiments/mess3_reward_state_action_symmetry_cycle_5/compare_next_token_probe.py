"""Compare frozen next-token probe KL with the existing affine belief MSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = STUDY_DIR / "next_token_probe" / "results"


def _condition_name(condition: dict[str, Any]) -> str:
    action = (
        "action_conditioned"
        if condition["condition_on_selected_action"]
        else "action_blind"
    )
    return f"context_{condition['context_len']}_{action}"


def load_joined_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Join probe KL and affine MSE at checkpoints where MSE was measured."""

    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/next_token_probe_curve.json")):
        payload = json.loads(path.read_text())
        seed = int(payload["protocol"]["seed"])
        for checkpoint in payload["checkpoints"]:
            belief = checkpoint.get("belief_probe")
            if belief is None:
                continue
            for condition in checkpoint["conditions"]:
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint_name": checkpoint["checkpoint_name"],
                        "agent_steps": int(checkpoint["agent_steps"]),
                        "condition": _condition_name(condition),
                        "belief_mse": float(belief["mse"]),
                        "belief_global_mse_ratio": float(
                            belief["global_mse_ratio"]
                        ),
                        "next_token_soft_kl_nats": float(
                            condition["soft_kl_nats"]
                        ),
                        "next_token_fraction_predictive_kl_removed": float(
                            condition["fraction_predictive_kl_removed"]
                        ),
                    }
                )
    return rows


def _correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([row["belief_mse"] for row in rows])
    y = np.asarray([row["next_token_soft_kl_nats"] for row in rows])
    if len(rows) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"n": len(rows), "pearson_r": None, "spearman_rho": None}
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": len(rows),
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({row["condition"] for row in rows})
    result: dict[str, Any] = {
        "schema_version": 1,
        "joined_row_count": len(rows),
        "metric_pair": {
            "x": "held-out affine belief MSE",
            "y": "exact-target next-token forward KL (nats)",
            "direction": "lower_is_better_for_both",
            "warning": (
                "The metrics test different information and should be compared "
                "by association, not numeric equality."
            ),
        },
        "conditions": {},
    }
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        seeds = sorted({row["seed"] for row in selected})
        result["conditions"][condition] = {
            "pooled": _correlation(selected),
            "per_seed": {
                str(seed): _correlation(
                    [row for row in selected if row["seed"] == seed]
                )
                for seed in seeds
            },
        }
    return result


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    conditions = sorted({row["condition"] for row in rows})
    if not conditions:
        raise ValueError("no joined probe rows to plot")
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    for axis, condition in zip(axes.flat, conditions):
        selected = [row for row in rows if row["condition"] == condition]
        for seed in sorted({row["seed"] for row in selected}):
            seed_rows = sorted(
                (row for row in selected if row["seed"] == seed),
                key=lambda row: row["agent_steps"],
            )
            axis.plot(
                [row["belief_mse"] for row in seed_rows],
                [row["next_token_soft_kl_nats"] for row in seed_rows],
                marker="o",
                linewidth=1,
                markersize=3,
                label=f"seed {seed}",
            )
        axis.set_title(condition.replace("_", " "))
        axis.set_xlabel("Affine belief MSE")
        axis.set_ylabel("Next-token KL (nats)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(conditions) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Cycle 5 variant 1: belief MSE vs next-token probe")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_DIR / "next_token_probe" / "comparison",
    )
    args = parser.parse_args()
    rows = load_joined_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"no completed probe curves under {args.results_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "joined_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "correlations.json").write_text(
        json.dumps(summarize(rows), indent=2, sort_keys=True) + "\n"
    )
    plot(rows, args.output_dir / "mse_vs_next_token_kl.png")


if __name__ == "__main__":
    main()
