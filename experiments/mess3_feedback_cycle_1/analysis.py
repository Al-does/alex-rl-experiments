"""Checkpoint probes that separate action-aware from action-blind beliefs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checkpoints import load_algorithm
from analysis.plots import simplex_scatter
from analysis.probes import (
    cluster_bootstrap_statistics,
    conditional_mse_metrics,
    fit_affine_probe,
    global_mse_metrics,
    held_out_permutation_null,
    mean_squared_error,
    percentile_interval,
    probe_predict,
    r2_score,
)
from experiments.mess3_feedback_cycle_1.dynamics import feedback_transitions
from experiments.mess3_feedback_cycle_1.probe import (
    FeedbackProbeData,
    branch_keys,
    collect_feedback_probe_data,
    make_feedback_filters,
    state_conditioned_guess_counts,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int


PROBE_RIDGE = 1e-6
MIN_GROUP_SIZE = 50
N_ENVS = 16
FULL_TRAIN_STEPS = 60_000
FULL_TEST_STEPS = 80_000
FULL_CALIBRATION_STEPS = 16_000
SMOKE_STEPS = 4_096
SMOKE_CALIBRATION_STEPS = 1_024
FULL_RESAMPLES = 1_000
SMOKE_RESAMPLES = 100
PERMUTATION_SAMPLE_CAP = 4_096
PLOT_SAMPLE_SIZE = 40_000
CONTEXT_LENGTH = 10
PRIMARY_TARGET = "executed"
SECONDARY_TARGETS = ("blind", "marginal", "joint", "factor_m", "factor_phi")
FACTOR_TARGETS = ("factor_m", "factor_phi")
# Below this target variance an affine probe has nothing to explain: the
# factor marginals collapse to their priors whenever the guess feedback is
# partial, because only their sum is ever observed.
MIN_TARGET_VARIANCE = 1e-6
VARIANCE_FRACTION = 0.95
_STREAM_KEYS = {
    "probe_calibration": (199,),
    "probe_train": (200,),
    "probe_test": (201,),
    "plot_sample": (202,),
    "bootstrap": (203,),
    "permutation": (204,),
    "permutation_sample": (205,),
}


@dataclass(frozen=True, slots=True)
class ProbeBudget:
    """Rollout and resampling sizes for one checkpoint probe."""

    calibration: int
    train: int
    test: int
    resamples: int

    @classmethod
    def for_context(cls, context: RunContext) -> ProbeBudget:
        if context.smoke:
            return cls(
                calibration=SMOKE_CALIBRATION_STEPS,
                train=SMOKE_STEPS,
                test=SMOKE_STEPS,
                resamples=SMOKE_RESAMPLES,
            )
        return cls(
            calibration=FULL_CALIBRATION_STEPS,
            train=FULL_TRAIN_STEPS,
            test=FULL_TEST_STEPS,
            resamples=FULL_RESAMPLES,
        )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    metrics: dict[str, Any]
    target_display: np.ndarray
    decoded_display: np.ndarray
    blind_display: np.ndarray
    factor_display: dict[str, tuple[np.ndarray, np.ndarray]]


def _device(context: RunContext) -> str:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _simplex_display(points: np.ndarray) -> np.ndarray:
    clipped = np.clip(points, 0.0, None)
    return clipped / np.maximum(clipped.sum(axis=1, keepdims=True), 1e-12)


def _episode_clusters(data: FeedbackProbeData) -> np.ndarray:
    """Build cluster IDs without treating correlated timesteps as independent."""

    clusters = np.empty(len(data.episode_steps), dtype=np.int64)
    next_cluster = 0
    for env_index in np.unique(data.env_indices):
        members = np.flatnonzero(data.env_indices == env_index)
        current_cluster = next_cluster
        first = True
        for index in members:
            if not first and data.episode_steps[index] == 0:
                current_cluster += 1
            clusters[index] = current_cluster
            first = False
        next_cluster = current_cluster + 1
    return clusters


def effective_dimension(
    activations: np.ndarray,
    *,
    variance_fraction: float = VARIANCE_FRACTION,
) -> dict[str, float]:
    """Count residual-stream directions needed to explain most of the variance."""

    centered = np.asarray(activations, dtype=np.float64)
    centered = centered - centered.mean(axis=0)
    spectrum = np.linalg.svdvals(centered) ** 2
    total = spectrum.sum()
    if total <= 0.0:
        return {"effective_dimension": float("nan"), "top_two_variance": float("nan")}
    cumulative = np.cumsum(spectrum) / total
    return {
        "effective_dimension": float(
            int(np.searchsorted(cumulative, variance_fraction) + 1)
        ),
        "top_two_variance": float(cumulative[1]) if len(cumulative) > 1 else 1.0,
        "variance_fraction": float(variance_fraction),
    }


def readout_subspace(weight: np.ndarray, *, rank: int) -> np.ndarray:
    """Return an orthonormal basis for a probe's dominant read-out directions."""

    matrix = np.asarray(weight, dtype=np.float64)
    left, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank]


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return normalized overlap: zero is orthogonal, one is coincident."""

    products = np.asarray(left).T @ np.asarray(right)
    return float(
        np.square(products).sum() / min(left.shape[1], right.shape[1])
    )


def _permutation_null_metrics(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    *,
    n_permutations: int,
    sample_seed: int,
    permutation_seed: int,
) -> dict[str, float | int]:
    """Evaluate a held-out shuffled-label null on fixed random subsets."""

    sample_rng = np.random.default_rng(sample_seed)
    n_train = min(PERMUTATION_SAMPLE_CAP, len(train_targets))
    n_test = min(PERMUTATION_SAMPLE_CAP, len(test_targets))
    train_indices = sample_rng.choice(len(train_targets), n_train, replace=False)
    test_indices = sample_rng.choice(len(test_targets), n_test, replace=False)
    features = train_features[train_indices]
    targets = train_targets[train_indices]
    held_out_features = test_features[test_indices]
    held_out_targets = test_targets[test_indices]

    def fit_predict(permuted: np.ndarray) -> np.ndarray:
        weight, bias = fit_affine_probe(features, permuted, ridge=PROBE_RIDGE)
        return probe_predict(weight, bias, held_out_features)

    null = held_out_permutation_null(
        targets,
        fit_predict,
        held_out_targets,
        n_permutations=n_permutations,
        seed=permutation_seed,
    )
    real = mean_squared_error(fit_predict(targets), held_out_targets)
    quantiles = np.quantile(null, [0.05, 0.5, 0.95])
    return {
        "permutation_real_mse": real,
        "permutation_null_mse_p05": float(quantiles[0]),
        "permutation_null_mse_p50": float(quantiles[1]),
        "permutation_null_mse_p95": float(quantiles[2]),
        "permutation_null_p_value_lower_tail": float(
            (1 + np.count_nonzero(null <= real)) / (len(null) + 1)
        ),
        "permutation_null_n": int(n_permutations),
    }


def _fit_target(
    train: FeedbackProbeData,
    test: FeedbackProbeData,
    name: str,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    """Fit one affine probe and report held-out error, or explain the skip."""

    train_targets = train.target(name)
    test_targets = test.target(name)
    if train_targets is None or test_targets is None:
        return {"target": name, "status": "unavailable"}, None, None
    variance = float(
        np.square(test_targets - test_targets.mean(axis=0)).mean()
    )
    if variance < MIN_TARGET_VARIANCE:
        return (
            {
                "target": name,
                "status": "degenerate",
                "target_variance": variance,
                "reason": "the target is constant, so affine decoding is vacuous",
            },
            None,
            None,
        )
    weight, bias = fit_affine_probe(
        train.activations,
        train_targets,
        ridge=PROBE_RIDGE,
    )
    predicted = probe_predict(weight, bias, test.activations)
    metrics: dict[str, Any] = {
        "target": name,
        "status": "fitted",
        **global_mse_metrics(predicted, test_targets),
        "r_squared": r2_score(predicted, test_targets),
    }
    return metrics, predicted, weight


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    feedback_strength: float,
    agent_steps: int | None = None,
    budget: ProbeBudget | None = None,
) -> ProbeResult:
    """Fit every action-awareness target on disjoint held-out rollouts."""

    if context.seed is None:
        raise ValueError("feedback probing requires a resolved seed")
    context.results_dir.mkdir(parents=True, exist_ok=True)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    budget = budget or ProbeBudget.for_context(context)
    train_steps = budget.train
    test_steps = budget.test
    calibration_steps = budget.calibration
    warmup = 4 if context.smoke else 64
    n_resamples = budget.resamples

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        environment_class = algorithm.config.env
        environment_config = dict(algorithm.config.env_config)
        environment_config["diagnostics"] = {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        }

        def make_environment():
            return environment_class(environment_config)

        environment = make_environment()
        try:
            filters = make_feedback_filters(
                environment,
                feedback_strength=feedback_strength,
            )
            base_transition = np.asarray(
                environment.model.transition_matrix,
                dtype=np.float64,
            )
            n_states = environment.model.n_states
            n_actions = int(environment.action_space.n)
        finally:
            environment.close()

        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": _device(context),
            "warmup": warmup,
            "n_envs": N_ENVS,
        }
        calibration = collect_feedback_probe_data(
            filters=filters,
            n_steps=calibration_steps,
            seed=streams["probe_calibration"],
            **common,
        )
        counts = state_conditioned_guess_counts(
            calibration,
            n_states=n_states,
            n_actions=n_actions,
        )
        guess_given_state = (counts + 1.0) / (counts + 1.0).sum(
            axis=1,
            keepdims=True,
        )
        marginal_transition = np.einsum(
            "sa,asj->sj",
            guess_given_state,
            feedback_transitions(
                feedback_strength,
                base=base_transition,
                n_actions=n_actions,
            ),
        )
        calibrated = filters.with_marginal(marginal_transition)
        train = collect_feedback_probe_data(
            filters=calibrated,
            n_steps=train_steps,
            seed=streams["probe_train"],
            **common,
        )
        test = collect_feedback_probe_data(
            filters=calibrated,
            n_steps=test_steps,
            seed=streams["probe_test"],
            **common,
        )

    target_error = max(
        float(np.max(np.abs(data.executed - data.diagnostic)))
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "the action-conditioned target disagrees with environment "
            f"diagnostics: {target_error:.3e}"
        )

    primary_metrics, primary_predicted, _ = _fit_target(train, test, PRIMARY_TARGET)
    if primary_predicted is None:
        raise RuntimeError("the action-conditioned belief target must be fittable")
    fine_metrics = conditional_mse_metrics(
        primary_predicted,
        test.executed,
        branch_keys(test, depth=2),
        min_group_size=MIN_GROUP_SIZE,
    )
    clusters = _episode_clusters(test)
    bootstrap = cluster_bootstrap_statistics(
        clusters,
        lambda indices: mean_squared_error(
            primary_predicted[indices],
            test.executed[indices],
        ),
        n_resamples=n_resamples,
        seed=seed_sequence_to_int(streams["bootstrap"], bits=32),
    )
    mse_ci_low, mse_ci_high = percentile_interval(bootstrap)
    permutation_metrics = _permutation_null_metrics(
        train.activations,
        train.executed,
        test.activations,
        test.executed,
        n_permutations=n_resamples,
        sample_seed=seed_sequence_to_int(streams["permutation_sample"], bits=32),
        permutation_seed=seed_sequence_to_int(streams["permutation"], bits=32),
    )

    targets: dict[str, Any] = {PRIMARY_TARGET: primary_metrics}
    predictions: dict[str, np.ndarray] = {PRIMARY_TARGET: primary_predicted}
    weights: dict[str, np.ndarray] = {}
    for name in SECONDARY_TARGETS:
        metrics, predicted, weight = _fit_target(train, test, name)
        targets[name] = metrics
        if predicted is not None:
            predictions[name] = predicted
        if weight is not None:
            weights[name] = weight

    geometry: dict[str, Any] = dict(effective_dimension(test.activations))
    if all(name in weights for name in FACTOR_TARGETS):
        bases = {
            name: readout_subspace(weights[name], rank=2)
            for name in FACTOR_TARGETS
        }
        geometry["factor_subspace_overlap"] = subspace_overlap(
            bases["factor_m"],
            bases["factor_phi"],
        )
    else:
        geometry["factor_subspace_overlap"] = float("nan")

    blind_ratio = targets["blind"].get("global_mse_ratio")
    executed_ratio = primary_metrics["global_mse_ratio"]
    metrics = {
        "condition": condition,
        "feedback_strength": float(feedback_strength),
        "checkpoint_step": agent_steps,
        "is_untrained": agent_steps == 0,
        "target": "exact_action_conditioned_predictive_belief",
        "probe": "held_out_affine_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "representation": "post_final_layer_norm",
        "sampling_distribution": "process_weighted_rollout",
        "policy_mode": "greedy",
        "warmup": warmup,
        "n_envs": N_ENVS,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "calibration_steps": calibration_steps,
        "branch_depth": 2,
        "min_group_size": MIN_GROUP_SIZE,
        **{
            key: value
            for key, value in primary_metrics.items()
            if key not in {"target", "status"}
        },
        **fine_metrics,
        "mse_ci_95_low": mse_ci_low,
        "mse_ci_95_high": mse_ci_high,
        "bootstrap_n": n_resamples,
        "bootstrap_cluster": "environment_episode",
        **permutation_metrics,
        "targets": targets,
        "geometry": geometry,
        "action_awareness_ratio": (
            float("nan")
            if blind_ratio is None or blind_ratio == 0.0
            else float(executed_ratio / blind_ratio)
        ),
        "action_blind_belief_mse": float(
            np.square(test.blind - test.executed).mean()
        ),
        "marginal_belief_mse": (
            float("nan")
            if test.marginal is None
            else float(np.square(test.marginal - test.executed).mean())
        ),
        "guess_given_state": guess_given_state.tolist(),
        "marginal_transition": marginal_transition.tolist(),
        **test.product_state_gap(),
        "token_accuracy_greedy": float(test.rewards.mean()),
        "n_fit": len(train.executed),
        "n_test": len(test.executed),
        "target_consistency_max_abs": target_error,
        "interpretation": (
            "Affine decodability does not establish causal policy use."
        ),
    }
    if agent_steps == 0:
        metrics["untrained_mse"] = metrics["mse"]

    sample_size = min(PLOT_SAMPLE_SIZE, len(test.executed))
    sample_rng = np.random.default_rng(streams["plot_sample"])
    indices = sample_rng.choice(len(test.executed), sample_size, replace=False)
    factor_display = {
        name: (
            test.target(name)[indices],
            _simplex_display(predictions[name][indices]),
        )
        for name in FACTOR_TARGETS
        if name in predictions
    }
    result = ProbeResult(
        metrics=metrics,
        target_display=test.executed[indices],
        decoded_display=_simplex_display(primary_predicted[indices]),
        blind_display=test.blind[indices],
        factor_display=factor_display,
    )
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    plot_probe_triplet(
        result,
        title=condition.replace("_", " "),
        path=context.results_dir / "belief_simplex.png",
    )
    if factor_display:
        plot_factor_geometry(
            result,
            title=condition.replace("_", " "),
            path=context.results_dir / "factor_geometry.png",
        )
    return result


def _draw_probe_triplet(
    axes: np.ndarray,
    result: ProbeResult,
    *,
    title: str,
) -> None:
    colors = np.clip(result.target_display, 0.0, 1.0)
    simplex_scatter(
        axes[0],
        result.target_display,
        colors=colors,
        s=0.25,
        alpha=0.18,
        title=f"{title}: action-conditioned Bayesian target",
        labels=("s0", "s1", "s2"),
    )
    simplex_scatter(
        axes[1],
        result.decoded_display,
        colors=colors,
        s=0.25,
        alpha=0.18,
        title=(
            f"{title}: affine-decoded residual stream\n"
            f"MSE={result.metrics['mse']:.5f}, "
            f"ratio={result.metrics['global_mse_ratio']:.3f}"
        ),
        labels=("s0", "s1", "s2"),
    )
    simplex_scatter(
        axes[2],
        result.blind_display,
        colors=colors,
        s=0.25,
        alpha=0.18,
        title=(
            f"{title}: action-blind belief\n"
            f"gap to target={result.metrics['action_blind_belief_mse']:.5f}"
        ),
        labels=("s0", "s1", "s2"),
    )


def plot_probe_triplet(result: ProbeResult, *, title: str, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    _draw_probe_triplet(axes, result, title=title)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_init_final(
    initial: ProbeResult,
    final: ProbeResult,
    *,
    condition: str,
    path: Path,
) -> None:
    """Compare exact, decoded, and action-blind simplexes before and after."""

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), squeeze=False)
    _draw_probe_triplet(axes[0], initial, title=f"{condition} - init")
    _draw_probe_triplet(axes[1], final, title=f"{condition} - final")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_factor_geometry(result: ProbeResult, *, title: str, path: Path) -> None:
    """Show both composition factors and their affine decodings."""

    names = [name for name in FACTOR_TARGETS if name in result.factor_display]
    figure, axes = plt.subplots(
        len(names),
        2,
        figsize=(9.2, 4.2 * len(names)),
        squeeze=False,
    )
    labels = {
        "factor_m": ("passive MESS3 factor", ("m0", "m1", "m2")),
        "factor_phi": ("guess-driven Z3 register", ("p0", "p1", "p2")),
    }
    overlap = result.metrics["geometry"]["factor_subspace_overlap"]
    for row, name in enumerate(names):
        target, decoded = result.factor_display[name]
        caption, vertices = labels[name]
        colors = np.clip(target, 0.0, 1.0)
        simplex_scatter(
            axes[row][0],
            target,
            colors=colors,
            s=0.25,
            alpha=0.18,
            title=f"{title}: {caption} target",
            labels=vertices,
        )
        simplex_scatter(
            axes[row][1],
            decoded,
            colors=colors,
            s=0.25,
            alpha=0.18,
            title=(
                f"{title}: decoded {caption}\n"
                f"ratio="
                f"{result.metrics['targets'][name]['global_mse_ratio']:.3f}, "
                f"subspace overlap={overlap:.3f}"
            ),
            labels=vertices,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_probe_trajectory(
    checkpoints: list[dict[str, Any]],
    *,
    condition: str,
    ceiling: float,
    path: Path,
) -> None:
    """Plot every target's normalized probe error beside task success."""

    steps = np.asarray(
        [point["agent_steps"] for point in checkpoints],
        dtype=np.float64,
    )
    success = 100.0 * np.asarray(
        [point["token_accuracy_greedy"] for point in checkpoints],
        dtype=np.float64,
    )
    figure, left = plt.subplots(figsize=(8.6, 5.0))
    right = left.twinx()
    styles = {
        "executed": ("#355c9a", "o", "Action-conditioned belief"),
        "blind": ("#7f9dc9", "^", "Action-blind belief"),
        "marginal": ("#5aa17f", "v", "Stacked single-HMM belief"),
    }
    for name, (color, marker, label) in styles.items():
        ratios = np.asarray(
            [
                point["targets"].get(name, {}).get("global_mse_ratio", np.nan)
                for point in checkpoints
            ],
            dtype=np.float64,
        )
        if np.all(np.isnan(ratios)):
            continue
        left.plot(steps, ratios, marker=marker, color=color, label=label)
    right.plot(
        steps,
        success,
        marker="s",
        color="#c45135",
        label="Greedy token success",
    )
    right.axhline(
        100.0 * ceiling,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label=f"Myopic Bayes ceiling ({100.0 * ceiling:.2f}%)",
    )
    left.set_xlabel("Environment steps")
    left.set_ylabel("Held-out probe MSE / target variance", color="#355c9a")
    right.set_ylabel("Task success (%)", color="#c45135")
    left.set_title(condition.replace("_", " "))
    # Reserve headroom so the combined legend never sits on top of a curve.
    left.set_ylim(top=left.get_ylim()[1] * 1.45)
    right.set_ylim(top=max(right.get_ylim()[1], 100.0 * ceiling) * 1.2)
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        fontsize=8,
    )
    left.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_contrast(
    summaries: dict[str, dict[str, Any]],
    *,
    path: Path,
) -> None:
    """Compare paired arms on how much their beliefs track their own guesses."""

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    palette = ("#355c9a", "#c45135", "#5aa17f", "#8a5ba8", "#b58a2b")
    for index, (name, summary) in enumerate(sorted(summaries.items())):
        color = palette[index % len(palette)]
        points = summary["checkpoint_probes"]
        steps = np.asarray([point["agent_steps"] for point in points])
        axes[0].plot(
            steps,
            [point["action_awareness_ratio"] for point in points],
            marker="o",
            color=color,
            label=name.replace("_", " "),
        )
        axes[1].plot(
            steps,
            [point["targets"]["executed"]["global_mse_ratio"] for point in points],
            marker="o",
            color=color,
            label=f"{name.replace('_', ' ')}: action conditioned",
        )
        axes[1].plot(
            steps,
            [point["targets"]["blind"]["global_mse_ratio"] for point in points],
            marker="^",
            linestyle="--",
            alpha=0.7,
            color=color,
            label=f"{name.replace('_', ' ')}: action blind",
        )
    axes[0].axhline(
        1.0,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="No preference for either target",
    )
    axes[0].set_ylabel("Action-conditioned MSE / action-blind MSE")
    axes[0].set_title("Does the residual stream track the agent's own guess?")
    axes[1].set_ylabel("Held-out probe MSE / target variance")
    axes[1].set_title("Both Bayesian targets, decoded from the same features")
    for axis in axes:
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def probe_at(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    feedback_strength: float,
    agent_steps: int,
    ceiling: float,
    budget: ProbeBudget | None = None,
) -> tuple[ProbeResult, dict[str, Any]]:
    """Probe one checkpoint and summarize it as one trajectory point."""

    probe_dir = context.results_dir / "checkpoint_probes" / (
        f"steps_{agent_steps:09d}"
    )
    result = probe_checkpoint(
        replace(context, results_dir=probe_dir, resume_from=checkpoint),
        checkpoint=checkpoint,
        condition=condition,
        feedback_strength=feedback_strength,
        agent_steps=agent_steps,
        budget=budget,
    )
    point = {
        "agent_steps": agent_steps,
        "mse": float(result.metrics["mse"]),
        "target_variance": float(result.metrics["target_variance"]),
        "global_mse_ratio": float(result.metrics["global_mse_ratio"]),
        "fine_mse_ratio": float(result.metrics["fine_mse_ratio"]),
        "r_squared": float(result.metrics["r_squared"]),
        "action_awareness_ratio": float(result.metrics["action_awareness_ratio"]),
        "action_blind_belief_mse": float(
            result.metrics["action_blind_belief_mse"]
        ),
        "marginal_belief_mse": float(result.metrics["marginal_belief_mse"]),
        "token_accuracy_greedy": float(result.metrics["token_accuracy_greedy"]),
        "myopic_ceiling": ceiling,
        "targets": {
            name: {
                key: value
                for key, value in payload.items()
                if key in {"status", "mse", "global_mse_ratio", "r_squared"}
            }
            for name, payload in result.metrics["targets"].items()
        },
        "geometry": result.metrics["geometry"],
        "probe": result.metrics,
    }
    return result, point
