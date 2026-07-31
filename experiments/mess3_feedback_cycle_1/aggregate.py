"""Aggregate the five feedback-cycle seed reports."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

from experiments.mess3_feedback_cycle_1.shared import DEFAULT_SEEDS


RUN_PREFIX = "mess3_feedback_cycle_1-ppo-seed"


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def aggregate(results_root: Path) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    curves: list[list[dict[str, Any]]] = []
    for seed in DEFAULT_SEEDS:
        run_dir = results_root / f"{RUN_PREFIX}{seed}-2m"
        summary = json.loads((run_dir / "condition_summary.json").read_text())
        curve = json.loads((run_dir / "checkpoint_probe_curve.json").read_text())[
            "checkpoints"
        ]
        final = summary["final_probe"]
        ablation = summary["causal_evaluations"]["action_input_ablation"]
        counterfactual = summary["causal_evaluations"][
            "counterfactual_belief_shift"
        ]
        per_seed.append(
            {
                "seed": seed,
                "run_id": run_dir.name,
                "final_steps": int(curve[-1]["agent_steps"]),
                "final_mse": float(final["mse"]),
                "final_global_mse_ratio": float(final["global_mse_ratio"]),
                "final_r_squared": float(final["r_squared"]),
                "token_accuracy": float(final["token_accuracy_greedy"]),
                "bayes_accuracy_on_rollout": float(
                    final["bayesian_optimal_accuracy_on_rollout"]
                ),
                "mask_delta_accuracy": float(
                    ablation["corruptions"]["mask"]["delta_token_accuracy"]
                ),
                "shuffle_delta_accuracy": float(
                    ablation["corruptions"]["shuffle"]["delta_token_accuracy"]
                ),
                "counterfactual_shift_mse": float(
                    counterfactual["shift_mse"]
                ),
                "counterfactual_shift_cosine": float(
                    counterfactual["shift_cosine_mean"]
                ),
            }
        )
        curves.append(curve)

    checkpoint_count = len(curves[0])
    if any(len(curve) != checkpoint_count for curve in curves):
        raise ValueError("all seed curves must have the same checkpoint count")
    aggregate_curve = []
    for checkpoint_index in range(checkpoint_count):
        points = [curve[checkpoint_index] for curve in curves]
        aggregate_curve.append(
            {
                "checkpoint_index": checkpoint_index,
                "agent_steps": _stats(
                    [float(point["agent_steps"]) for point in points]
                ),
                "mse": _stats([float(point["mse"]) for point in points]),
                "global_mse_ratio": _stats(
                    [float(point["global_mse_ratio"]) for point in points]
                ),
                "token_accuracy": _stats(
                    [float(point["token_accuracy_greedy"]) for point in points]
                ),
            }
        )
    metric_names = [
        "final_mse",
        "final_global_mse_ratio",
        "final_r_squared",
        "token_accuracy",
        "bayes_accuracy_on_rollout",
        "mask_delta_accuracy",
        "shuffle_delta_accuracy",
        "counterfactual_shift_mse",
        "counterfactual_shift_cosine",
    ]
    return {
        "study": "mess3_feedback_cycle_1",
        "condition": "ppo_feedback_eta_0p10",
        "seeds": list(DEFAULT_SEEDS),
        "per_seed": per_seed,
        "aggregate": {
            name: _stats([float(run[name]) for run in per_seed])
            for name in metric_names
        },
        "mse_over_training": aggregate_curve,
    }


def render_findings(payload: dict[str, Any]) -> str:
    aggregate_metrics = payload["aggregate"]
    return "\n".join(
        [
            "# MESS3 feedback cycle 1 — five-seed findings",
            "",
            "Plain PPO with `eta=0.10`, `delay=1`, previous-action input, and "
            "2M environment steps per seed.",
            "",
            "| metric | mean | sample std |",
            "|---|---:|---:|",
            *[
                f"| {name} | {values['mean']:.6g} | {values['stdev']:.6g} |"
                for name, values in aggregate_metrics.items()
            ],
            "",
            "See `five_seed_summary.json` for per-seed values and the full "
            "MSE-over-training aggregate.",
            "",
        ]
    )


def write_summary(results_root: Path) -> Path:
    payload = aggregate(results_root)
    output = results_root / "mess3-feedback-c1-five-seed-summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "five_seed_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (output / "findings.md").write_text(render_findings(payload))
    return output


if __name__ == "__main__":
    write_summary(Path(__file__).resolve().parent / "ppo" / "results")
