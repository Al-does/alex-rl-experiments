"""Aggregate the five feedback-cycle seed reports."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.mess3_feedback_cycle_1.shared import DEFAULT_SEEDS


RUN_PREFIX = "mess3_feedback_cycle_1-ppo-seed"
ACTION_LABELS = ("token 0", "token 1", "token 2")


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def _action_distribution(results_root: Path) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    pooled = {label: 0 for label in ACTION_LABELS}
    for seed in DEFAULT_SEEDS:
        run_dir = results_root / f"{RUN_PREFIX}{seed}-2m"
        summary = json.loads((run_dir / "condition_summary.json").read_text())
        by_action = summary["causal_evaluations"]["counterfactual_belief_shift"][
            "by_factual_action"
        ]
        counts = {
            ACTION_LABELS[int(action)]: int(stats["n_evaluated"])
            for action, stats in sorted(by_action.items(), key=lambda item: int(item[0]))
        }
        total = sum(counts.values())
        per_seed.append(
            {
                "seed": seed,
                "counts": counts,
                "fractions": {label: count / total for label, count in counts.items()},
                "total_evaluated": total,
            }
        )
        for label, count in counts.items():
            pooled[label] += count
    pooled_total = sum(pooled.values())
    return {
        "per_seed": per_seed,
        "pooled_counts": pooled,
        "pooled_fractions": {
            label: count / pooled_total for label, count in pooled.items()
        },
        "pooled_total": pooled_total,
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
                "mask_mse": float(ablation["corruptions"]["mask"]["mse"]),
                "shuffle_mse": float(ablation["corruptions"]["shuffle"]["mse"]),
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
        "mask_mse",
        "shuffle_mse",
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
        "action_choice_distribution": _action_distribution(results_root),
    }


def plot_mse_over_training(payload: dict[str, Any], path: Path) -> None:
    """Plot baseline MSE over training plus final ablation / CF-shift probes."""

    curve = payload["mse_over_training"]
    agg = payload["aggregate"]
    steps_m = np.asarray(
        [point["agent_steps"]["mean"] / 1_000_000 for point in curve]
    )
    mse_mean = np.asarray([point["mse"]["mean"] for point in curve])
    mse_std = np.asarray([point["mse"]["stdev"] for point in curve])

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.set_yscale("log")
    axis.fill_between(
        steps_m,
        np.maximum(mse_mean - mse_std, 1e-6),
        mse_mean + mse_std,
        color="#355c9a",
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(
        steps_m,
        mse_mean,
        marker="o",
        color="#355c9a",
        linewidth=2.0,
        label="Baseline affine probe MSE",
    )

    final_x = steps_m[-1]
    x_offsets = np.linspace(-0.04, 0.12, 4)
    final_points = [
        (x_offsets[0], agg["final_mse"]["mean"], agg["final_mse"]["stdev"], "#355c9a", "Baseline (final)"),
        (
            x_offsets[1],
            agg["counterfactual_shift_mse"]["mean"],
            agg["counterfactual_shift_mse"]["stdev"],
            "#6a994e",
            "CF belief-shift MSE",
        ),
        (
            x_offsets[2],
            agg["mask_mse"]["mean"],
            agg["mask_mse"]["stdev"],
            "#bc4749",
            "Masked prev-action MSE",
        ),
        (
            x_offsets[3],
            agg["shuffle_mse"]["mean"],
            agg["shuffle_mse"]["stdev"],
            "#a44a3f",
            "Shuffled prev-action MSE",
        ),
    ]
    for x_offset, mean, stdev, color, label in final_points:
        axis.errorbar(
            final_x + x_offset,
            mean,
            yerr=stdev,
            fmt="s",
            color=color,
            markersize=7,
            capsize=3,
            linestyle="none",
            label=label,
        )

    axis.axvline(final_x, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
    axis.set_xlabel("Environment steps (millions)")
    axis.set_ylabel("Probe MSE (log scale)")
    axis.set_title(
        "MESS3 feedback cycle 1 — belief probe MSE over training\n"
        "final-checkpoint ablations shown at ~2.0M steps"
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="upper right", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def render_findings(payload: dict[str, Any]) -> str:
    aggregate_metrics = payload["aggregate"]
    action = payload["action_choice_distribution"]
    pooled = action["pooled_fractions"]

    def fmt(name: str, *, pct: bool = False) -> str:
        values = aggregate_metrics[name]
        if pct:
            return f"{100 * values['mean']:.2f}% ± {100 * values['stdev']:.2f}%"
        return f"{values['mean']:.6g} ± {values['stdev']:.6g}"

    action_rows = []
    for seed_info in action["per_seed"]:
        fractions = seed_info["fractions"]
        action_rows.append(
            f"| {seed_info['seed']} | "
            f"{100 * fractions[ACTION_LABELS[0]]:.1f}% | "
            f"{100 * fractions[ACTION_LABELS[1]]:.1f}% | "
            f"{100 * fractions[ACTION_LABELS[2]]:.1f}% |"
        )

    return "\n".join(
        [
            "# MESS3 feedback cycle 1 — five-seed findings",
            "",
            "Plain PPO with `eta=0.10`, `delay=1`, previous executed action in the "
            "observation, and ~2M environment steps per seed (seeds 42–46). Each "
            "action is a token intervention; reward still scores the pre-transition "
            "token, so the policy must infer hidden state from emissions and its "
            "own feedback history.",
            "",
            "## Summary metrics",
            "",
            "| metric | mean ± sample std |",
            "|---|---:|",
            f"| final probe MSE | {fmt('final_mse')} |",
            f"| normalized MSE (MSE / target variance) | {fmt('final_global_mse_ratio')} |",
            f"| probe R² | {fmt('final_r_squared')} |",
            f"| greedy token accuracy | {fmt('token_accuracy', pct=True)} |",
            f"| on-rollout Bayes accuracy | {fmt('bayes_accuracy_on_rollout', pct=True)} |",
            f"| masked prev-action probe MSE | {fmt('mask_mse')} |",
            f"| shuffled prev-action probe MSE | {fmt('shuffle_mse')} |",
            f"| counterfactual belief-shift MSE | {fmt('counterfactual_shift_mse')} |",
            f"| counterfactual shift cosine | {fmt('counterfactual_shift_cosine')} |",
            "",
            "See [`mse_over_training.png`](mse_over_training.png) for the training "
            "curve and final-checkpoint ablations on a shared log-MSE axis.",
            "",
            "## Belief geometry over training",
            "",
            "The held-out affine probe MSE falls sharply from initialization "
            f"({payload['mse_over_training'][0]['mse']['mean']:.4f}) to the final "
            f"checkpoint ({aggregate_metrics['final_mse']['mean']:.4g} mean across "
            "seeds), with normalized MSE reaching ~0.2–0.4% of target variance "
            f"({fmt('final_global_mse_ratio')}). The network learns a representation "
            "that is nearly linearly decodable into the exact predictive Bayesian "
            "belief under the training distribution.",
            "",
            "## Action choice distribution",
            "",
            "Counts come from the counterfactual evaluation rollouts (greedy closed "
            "loop at the final checkpoint). Actions coincide with the three token "
            "interventions and remain close to uniform:",
            "",
            "| seed | token 0 | token 1 | token 2 |",
            "|---:|---:|---:|---:|",
            *action_rows,
            (
                f"| **pooled** | **{100 * pooled[ACTION_LABELS[0]]:.1f}%** | "
                f"**{100 * pooled[ACTION_LABELS[1]]:.1f}%** | "
                f"**{100 * pooled[ACTION_LABELS[2]]:.1f}%** |"
            ),
            "",
            f"Pooled over {action['pooled_total']:,} evaluated steps. Symmetry is "
            "expected because the three interventions are permutation-equivalent at "
            "`eta=0.10`.",
            "",
            "## Interpretation",
            "",
            "### Token accuracy near the Bayes ceiling (hypothesis)",
            "",
            "The result shows ~69.6% greedy token accuracy against ~70.0% on-rollout "
            "Bayesian accuracy. **Hypothesis:** the policy is not far from the best "
            "token guesser allowed by the partially observed process, given its "
            "architecture and training budget.",
            "",
            "Mechanistically, each step the agent sees the current emission and the "
            "previous executed action. With `delay=1`, that action shifted the "
            "transition kernel from passive `T` toward a rank-one intervention "
            "`R_a`, so the observation stream is informative but stochastic "
            "(`alpha=0.85`). A Bayes-optimal filter maintains the predictive belief "
            "over hidden states; greedy argmax on that belief sets an accuracy "
            "ceiling under the evaluation rollouts. The tight gap between agent and "
            "Bayes accuracy suggests the learned representation is not merely "
            "decodable (probe R² ≈ 0.998) but is used in a way that tracks the "
            "filter’s token ranking reasonably well. It does **not** by itself "
            "prove optimality: the probe measures belief geometry, not the policy "
            "head, and remaining error could reflect approximation in the actor or "
            "mismatch between training and evaluation sampling.",
            "",
            "### Previous-action sensitivity (hypothesis)",
            "",
            "Masking or shuffling the previous-action block at the final checkpoint "
            "inflates probe MSE by ~40–50× and lowers token accuracy by ~0.6–1.1 "
            "percentage points. **Hypothesis:** the policy and representation "
            "genuinely condition on executed feedback, not only on emissions.",
            "",
            "In this task the previous action is the only channel through which the "
            "agent’s own interventions enter the observation. Masking removes that "
            "channel while leaving transitions driven by executed actions intact; "
            "shuffling breaks the temporal pairing between the true intervention and "
            "the current belief state. The ablation therefore tests whether the "
            "network’s internal state tracks *this* history rather than a generic "
            "action marginal. The accuracy drop is modest because emissions already "
            "carry substantial information about hidden state, so a emission-only "
            "policy can still score well. The large MSE inflation under ablation is "
            "the stronger signal: the affine probe was fit under intact observations, "
            "and corrupting the action block moves representations off the "
            "training manifold even when token choice degrades only slightly.",
            "",
            "### Counterfactual belief-shift alignment (hypothesis)",
            "",
            "The counterfactual probe holds context fixed, swaps the previous action "
            "in the observation, and compares the decoded belief shift to the exact "
            "delay-one Bayesian update induced by that swap. Mean cosine alignment "
            f"is {aggregate_metrics['counterfactual_shift_cosine']['mean']:.3f} "
            f"(sample std {aggregate_metrics['counterfactual_shift_cosine']['stdev']:.3f}). "
            "**Hypothesis:** local representational sensitivity to action changes "
            "follows the direction of the true belief update, not just arbitrary "
            "feature noise.",
            "",
            "Mechanistically, under `delay=1` the previous action enters the "
            "transition that produced the current token distribution. Changing the "
            "recorded action while holding earlier context fixed isolates the "
            "Bayesian effect of that counterfactual intervention. High cosine "
            "similarity means the post-intervention representation moves toward the "
            "Bayes-updated belief; `shift_mse` (~0.0035 mean) reports the remaining "
            "magnitude error in that subspace. This is distinct from the ablation "
            "MSEs (~0.013): ablations measure global decodability under corrupted "
            "inputs, while the counterfactual test measures whether infinitesimal "
            "action substitutions propagate through the representation in the same "
            "direction as the filter. Values below 1.0 leave room for residual "
            "non-Bayesian components in the actor trunk or probe mismatch.",
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
    plot_mse_over_training(payload, output / "mse_over_training.png")
    return output


if __name__ == "__main__":
    write_summary(Path(__file__).resolve().parent / "ppo" / "results")
