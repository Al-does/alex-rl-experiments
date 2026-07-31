"""Post-campaign analysis the per-run summaries cannot provide.

Two gaps in `campaign_summary.json`:

1. **Cross-arm comparison needs normalization.** Raw probe MSE is not
   comparable between arms because the target variance differs roughly
   threefold across the `epsilon` sweep. The raw ranking inverts once
   normalized, so every cross-arm claim must use `global_mse_ratio`
   (`= 1 - R^2`) rather than `mse`.

2. **The probe metrics need calibration bars.** The probe is affine, so it is
   not neutral between representations that carry the same information in
   different formats. In particular the joint belief is exactly the outer
   product of its factor marginals whenever `epsilon = 0`, and no affine map
   can recover a product from a representation that stores the factors
   separately. To find out what `action_awareness_ratio` and the per-target
   R-squared can actually report, we fit the study's own probe from hand-built
   idealized representations and read off the resulting scores.

The reference policy for calibration is the exact-filter myopic Bayes agent, so
its accuracy is a slight upper bound on the context-ten `myopic_ceiling`
recorded during training.

Run as a module: the study's local `analysis.py` shadows the harness `analysis`
package when this file is invoked by path.

```bash
uv run python -m experiments.mess3_feedback_factoring_cycle_1.campaign_analysis \
    --campaign 20260731T190000Z-eight-arms-five-seeds
```
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from analysis.probes import predictive_belief_update
from envs.hmm.env import HMMEnv
from experiments.mess3_feedback_factoring_cycle_1.dynamics import (
    chain_factor,
    composite_state_belief,
    factor_marginals,
    joint_transitions,
)
from experiments.mess3_feedback_factoring_cycle_1.shared import (
    CONDITIONS,
    Condition,
    condition_by_name,
    env_config,
)

STUDY = "mess3_feedback_factoring_cycle_1"
ROOT = Path(__file__).parent
SEEDS = (42, 43, 44, 45, 46)
TARGETS = (
    "joint",
    "blind",
    "marginal",
    "composite",
    "composite_blind",
    "factor_m",
    "factor_phi",
)
#: Late-training accuracy loss (relative to the run's own peak) that marks a
#: PPO collapse rather than ordinary checkpoint noise.
COLLAPSE_THRESHOLD = 0.02
PROBE_RIDGE = 1e-6
CALIBRATION_STEPS = 24_000
CALIBRATION_WARMUP = 64


# ---------------------------------------------------------------------------
# Normalized campaign summary
# ---------------------------------------------------------------------------


def run_summary_path(arm: str, seed: int) -> Path:
    return (
        ROOT / arm / "results" / f"{STUDY}-{arm}-seed{seed}" / "condition_summary.json"
    )


def load_runs(arm: str, seeds: tuple[int, ...] = SEEDS) -> list[dict[str, Any]]:
    """Load every available per-seed summary for one arm."""

    runs = []
    for seed in seeds:
        path = run_summary_path(arm, seed)
        if path.exists():
            runs.append({"seed": seed, **json.loads(path.read_text())})
    return runs


def _aggregate(values: list[float | None]) -> dict[str, float | int] | None:
    clean = [float(v) for v in values if v is not None and v == v]
    if not clean:
        return None
    return {
        "mean": statistics.mean(clean),
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }


def _target_ratio(checkpoint: dict[str, Any], target: str) -> float | None:
    """Return one target's normalized error, or ``None`` when not fitted."""

    entry = checkpoint.get("targets", {}).get(target, {})
    if entry.get("status") != "fitted":
        return None
    return entry.get("global_mse_ratio")


def detect_collapse(run: dict[str, Any]) -> dict[str, Any] | None:
    """Flag runs whose final accuracy sits well below their own peak.

    Three runs in this campaign lose accuracy in the last checkpoints. They
    inflate the arm standard deviations enough to change which arm looks best,
    so they are reported rather than averaged away.
    """

    accuracies = [
        point["token_accuracy_greedy"] for point in run["checkpoint_probes"]
    ]
    peak = max(accuracies)
    drop = peak - accuracies[-1]
    if drop <= COLLAPSE_THRESHOLD:
        return None
    return {
        "seed": run["seed"],
        "peak_accuracy": peak,
        "final_accuracy": accuracies[-1],
        "drop": drop,
        "trajectory": accuracies,
    }


def normalized_summary(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Summarize every arm on normalized error plus ceiling-relative success."""

    arms: dict[str, Any] = {}
    for condition in CONDITIONS:
        runs = load_runs(condition.name, seeds)
        if not runs:
            continue
        first = [run["checkpoint_probes"][0] for run in runs]
        last = [run["checkpoint_probes"][-1] for run in runs]
        collapses = [c for run in runs if (c := detect_collapse(run))]
        collapsed_seeds = {c["seed"] for c in collapses}
        healthy = [
            point["token_accuracy_greedy"]
            for run, point in zip(runs, last)
            if run["seed"] not in collapsed_seeds
        ]
        ceiling = last[0].get("myopic_ceiling")
        accuracy = _aggregate([p["token_accuracy_greedy"] for p in last])
        healthy_accuracy = _aggregate(healthy)

        targets: dict[str, Any] = {}
        for target in TARGETS:
            initial = _aggregate([_target_ratio(p, target) for p in first])
            final = _aggregate([_target_ratio(p, target) for p in last])
            targets[target] = {
                "normalized_error_init": initial,
                "normalized_error_final": final,
                "status": "degenerate" if final is None else "fitted",
                "change": (
                    None
                    if initial is None or final is None
                    else final["mean"] - initial["mean"]
                ),
            }

        arms[condition.name] = {
            "feedback_strength": condition.feedback_strength,
            "register_noise": condition.register_noise,
            "observe_previous_guess": condition.observe_previous_guess,
            "myopic_ceiling": ceiling,
            "token_accuracy_final": accuracy,
            "token_accuracy_final_excluding_collapses": healthy_accuracy,
            "fraction_of_ceiling": (
                None if not ceiling or accuracy is None else accuracy["mean"] / ceiling
            ),
            "fraction_of_ceiling_excluding_collapses": (
                None
                if not ceiling or healthy_accuracy is None
                else healthy_accuracy["mean"] / ceiling
            ),
            # Raw MSE is retained only so the normalization stays reconstructible.
            "raw_mse_final": _aggregate([p["mse"] for p in last]),
            "target_variance_final": _aggregate([p["target_variance"] for p in last]),
            "action_awareness_ratio_init": _aggregate(
                [p.get("action_awareness_ratio") for p in first]
            ),
            "action_awareness_ratio_final": _aggregate(
                [p.get("action_awareness_ratio") for p in last]
            ),
            "factor_subspace_overlap_init": _aggregate(
                [p["geometry"].get("factor_subspace_overlap") for p in first]
            ),
            "factor_subspace_overlap_final": _aggregate(
                [p["geometry"].get("factor_subspace_overlap") for p in last]
            ),
            "register_entropy_final": _aggregate(
                [p.get("register_entropy_nats") for p in last]
            ),
            "marginal_belief_mse_final": _aggregate(
                [p.get("marginal_belief_mse") for p in last]
            ),
            "marginal_belief_mse_normalized": _aggregate(
                [
                    None
                    if p.get("marginal_belief_mse") is None
                    else p["marginal_belief_mse"] / p["target_variance"]
                    for p in last
                ]
            ),
            "permutation_null_p_value": _aggregate(
                [
                    run["final_probe"].get("permutation_null_p_value_lower_tail")
                    for run in runs
                ]
            ),
            "targets": targets,
            "collapsed_runs": collapses,
        }
    return {
        "study": STUDY,
        "seeds": list(seeds),
        "metric": "global_mse_ratio (= 1 - R^2); lower is better, 1.0 = no better than the mean",
        "warning": (
            "Raw MSE is not comparable across arms: target variance differs "
            "about threefold across the epsilon sweep and the raw ranking "
            "inverts under normalization."
        ),
        "arms": arms,
    }


# ---------------------------------------------------------------------------
# Network-free metric calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferenceRollout:
    """Exact-filter targets for one condition, with no network involved."""

    joint: np.ndarray
    blind: np.ndarray
    composite: np.ndarray
    composite_blind: np.ndarray
    factor_m: np.ndarray
    factor_phi: np.ndarray
    accuracy: float

    def target(self, name: str) -> np.ndarray:
        return getattr(self, name)


def _diagnostic_env(condition: Condition) -> HMMEnv:
    config = dict(env_config(condition))
    config["diagnostics"] = {
        "state": True,
        "belief": True,
        "tokens": True,
        "transitions": True,
    }
    return HMMEnv(config)


def reference_rollout(
    condition: Condition,
    *,
    seed: int,
    n_steps: int = CALIBRATION_STEPS,
) -> ReferenceRollout:
    """Drive the real environment with an exact-filter myopic Bayes policy.

    This rebuilds the study's `joint`, `blind`, `composite` and factor targets
    without loading a checkpoint, which is what makes the calibration cheap.
    """

    env = _diagnostic_env(condition)
    model = env.model
    n_guesses = int(round(np.sqrt(model.n_states)))
    transitions = joint_transitions(
        condition.feedback_strength,
        base=chain_factor(model.transition_matrix, size=n_guesses),
        n_actions=n_guesses,
    )
    blind_transition = transitions.mean(axis=0)
    emission = np.asarray(model.emission_matrix, dtype=np.float64)
    initial = np.asarray(model.initial_distribution, dtype=np.float64)

    # P(scored composite token | state): aggregate the paired emission onto x.
    scored_likelihood = np.zeros((model.n_states, n_guesses))
    for token in range(model.n_tokens):
        scored_likelihood[:, token // n_guesses] += emission[:, token]

    joint = initial.copy()
    blind = initial.copy()
    collected: list[dict[str, Any]] = []

    env.reset(seed=seed)
    step = 0
    while len(collected) < n_steps:
        action = int(np.argmax(joint @ scored_likelihood))
        _obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        token = info.get("visible_token_current")
        if token is not None:
            measurement = np.diag(emission[:, int(token)])
            executed = np.asarray(
                info["executed_transition_matrix"], dtype=np.float64
            )
            joint = predictive_belief_update(joint, measurement @ executed)
            blind = predictive_belief_update(
                blind, measurement @ blind_transition
            )
            if step > CALIBRATION_WARMUP:
                chain, register = factor_marginals(joint[None, :])
                collected.append(
                    {
                        "joint": joint.copy(),
                        "blind": blind.copy(),
                        "composite": composite_state_belief(joint[None, :])[0],
                        "composite_blind": composite_state_belief(blind[None, :])[0],
                        "factor_m": chain[0],
                        "factor_phi": register[0],
                        "reward": reward,
                    }
                )

        if terminated or truncated:
            env.reset()
            joint = initial.copy()
            blind = initial.copy()
    env.close()

    def stack(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in collected], dtype=np.float64)

    return ReferenceRollout(
        joint=stack("joint"),
        blind=stack("blind"),
        composite=stack("composite"),
        composite_blind=stack("composite_blind"),
        factor_m=stack("factor_m"),
        factor_phi=stack("factor_phi"),
        accuracy=float(np.mean([row["reward"] for row in collected])),
    )


def affine_normalized_error(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
) -> float:
    """Fit the study's affine ridge probe and return `1 - R^2` held out."""

    design = np.concatenate(
        [train_features, np.ones((len(train_features), 1))], axis=1
    )
    gram = design.T @ design + PROBE_RIDGE * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ train_target)
    held_out = np.concatenate(
        [test_features, np.ones((len(test_features), 1))], axis=1
    )
    predicted = held_out @ coefficients
    mse = float(np.square(predicted - test_target).mean())
    variance = float(np.square(test_target - test_target.mean(axis=0)).mean())
    return float("nan") if variance == 0.0 else mse / variance


#: Hand-built representations spanning the hypotheses the study wants to
#: separate. `factored` is what the Factored World Hypothesis predicts;
#: `composite` is the reward-sufficient statistic under `gamma = 0`.
IDEALIZED: dict[str, Callable[[ReferenceRollout], np.ndarray]] = {
    "joint": lambda r: r.joint,
    "factored": lambda r: np.concatenate([r.factor_m, r.factor_phi], axis=1),
    "composite": lambda r: r.composite,
    "composite_plus_register": lambda r: np.concatenate(
        [r.composite, r.factor_phi], axis=1
    ),
    "composite_plus_factored": lambda r: np.concatenate(
        [r.composite, r.factor_m, r.factor_phi], axis=1
    ),
    "blind": lambda r: r.blind,
}
CALIBRATION_TARGETS = ("joint", "blind", "composite", "factor_m", "factor_phi")


def calibrate(
    arm: str,
    *,
    train_seed: int = 7,
    test_seed: int = 99,
    n_steps: int = CALIBRATION_STEPS,
) -> dict[str, Any]:
    """Score the study's probe metrics for each idealized representation."""

    condition = condition_by_name(arm)
    train = reference_rollout(condition, seed=train_seed, n_steps=n_steps)
    test = reference_rollout(condition, seed=test_seed, n_steps=n_steps)

    results: dict[str, Any] = {}
    for label, build in IDEALIZED.items():
        train_features, test_features = build(train), build(test)
        ratios: dict[str, float | None] = {}
        for target in CALIBRATION_TARGETS:
            observed = test.target(target)
            variance = float(np.square(observed - observed.mean(axis=0)).mean())
            if variance < 1e-12:
                ratios[target] = None
                continue
            ratios[target] = affine_normalized_error(
                train_features,
                train.target(target),
                test_features,
                observed,
            )
        blind_ratio = ratios.get("blind")
        joint_ratio = ratios.get("joint")
        results[label] = {
            "normalized_error": ratios,
            "action_awareness_ratio": (
                None
                if not blind_ratio or joint_ratio is None
                else joint_ratio / blind_ratio
            ),
            "n_features": int(build(test).shape[1]),
        }
    return {
        "arm": arm,
        "feedback_strength": condition.feedback_strength,
        "register_noise": condition.register_noise,
        "exact_filter_bayes_accuracy": train.accuracy,
        "representations": results,
    }


CALIBRATION_ARMS = (
    "factoring_free",
    "factoring_cheap",
    "factoring_costly",
    "factoring_impossible",
    "deterministic_feedback",
)


def calibration_report(
    arms: tuple[str, ...] = CALIBRATION_ARMS,
    *,
    n_steps: int = CALIBRATION_STEPS,
) -> dict[str, Any]:
    """Calibrate every arm where the factor targets are non-degenerate."""

    return {
        "study": STUDY,
        "probe": "held_out_affine_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "reference_policy": "exact_filter_myopic_bayes",
        "n_steps_per_split": n_steps,
        "interpretation": (
            "A perfectly factored representation is what the Factored World "
            "Hypothesis predicts. Where its action_awareness_ratio exceeds "
            "one, the study's prediction that training drives that ratio below "
            "one cannot be satisfied by a factored representation, so the "
            "metric cannot test the hypothesis at that operating point."
        ),
        "arms": {arm: calibrate(arm, n_steps=n_steps) for arm in arms},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        default="20260731T190000Z-eight-arms-five-seeds",
        help="campaign label under results/ to write into",
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=CALIBRATION_STEPS,
        help="reference-rollout length per train/test split",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="only rewrite the normalized summary",
    )
    arguments = parser.parse_args()

    destination = ROOT / "results" / arguments.campaign
    destination.mkdir(parents=True, exist_ok=True)

    summary = normalized_summary()
    (destination / "normalized_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"wrote {destination / 'normalized_summary.json'}")

    if not arguments.skip_calibration:
        report = calibration_report(n_steps=arguments.calibration_steps)
        (destination / "metric_calibration.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"wrote {destination / 'metric_calibration.json'}")


if __name__ == "__main__":
    main()
