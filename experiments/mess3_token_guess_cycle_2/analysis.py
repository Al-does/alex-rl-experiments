"""Checkpoint belief probes, simplex figures, and task-success curves."""

from __future__ import annotations

from dataclasses import dataclass
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
from envs.mess3.model import passive_model
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    branch_keys,
    collect_probe_data,
    make_transducer_target,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int


PROBE_RANK = 2
PROBE_RIDGE = 1e-6
MIN_GROUP_SIZE = 50
N_ENVS = 16
FULL_RESAMPLES = 1_000
SMOKE_RESAMPLES = 100
PERMUTATION_SAMPLE_CAP = 4_096
FULL_TEST_STEPS = 80_000
PLOT_SAMPLE_SIZE = 80_000
CONTEXT_LENGTH = 10
_STREAM_KEYS = {
    "probe_train": (200,),
    "probe_test": (201,),
    "plot_sample": (202,),
    "bootstrap": (203,),
    "permutation": (204,),
    "permutation_sample": (205,),
}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    metrics: dict[str, Any]
    target_display: np.ndarray
    decoded_display: np.ndarray


def bayesian_optimal_accuracy(
    *,
    context_length: int = CONTEXT_LENGTH,
    alpha: float = 0.85,
) -> float:
    """Exact finite-context Bayes classifier accuracy under stationarity."""

    if context_length < 0:
        raise ValueError("context_length must be non-negative")
    model = passive_model(alpha=alpha)
    transition = np.asarray(model.transition_matrix, dtype=np.float64)
    emission = np.asarray(model.emission_matrix, dtype=np.float64)
    weighted_histories = np.full((1, model.n_states), 1.0 / model.n_states)
    for _ in range(context_length):
        children = [
            weighted_histories @ np.diag(emission[:, token]) @ transition
            for token in range(model.n_tokens)
        ]
        weighted_histories = np.concatenate(children, axis=0)
    predictions = weighted_histories @ emission
    return float(predictions.max(axis=1).sum())


BAYESIAN_OPTIMAL_ACCURACY = bayesian_optimal_accuracy()


def fit_reduced_rank_affine(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an affine least-squares map constrained to an output rank."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("features and targets must be aligned matrices")
    if not 0 < rank <= min(x.shape[1], y.shape[1]):
        raise ValueError("rank is incompatible with feature and target dimensions")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    ordinary_weight, _, _, _ = np.linalg.lstsq(
        x_centered,
        y_centered,
        rcond=None,
    )
    fitted = x_centered @ ordinary_weight
    _, _, right = np.linalg.svd(fitted, full_matrices=False)
    output_subspace = right[:rank].T
    weight = ordinary_weight @ output_subspace @ output_subspace.T
    bias = y_mean - x_mean @ weight
    return weight, bias


def _device(context: RunContext) -> str:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _simplex_display(points: np.ndarray) -> np.ndarray:
    clipped = np.clip(points, 0.0, None)
    return clipped / np.maximum(clipped.sum(axis=1, keepdims=True), 1e-12)


def _episode_clusters(data: ProbeData) -> np.ndarray:
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


def _permutation_null_metrics(
    train: ProbeData,
    test: ProbeData,
    *,
    n_permutations: int,
    sample_seed: int,
    permutation_seed: int,
) -> dict[str, float | int]:
    """Evaluate a held-out shuffled-label null on fixed random subsets."""

    sample_rng = np.random.default_rng(sample_seed)
    n_train = min(PERMUTATION_SAMPLE_CAP, len(train.beliefs))
    n_test = min(PERMUTATION_SAMPLE_CAP, len(test.beliefs))
    train_indices = sample_rng.choice(
        len(train.beliefs),
        n_train,
        replace=False,
    )
    test_indices = sample_rng.choice(
        len(test.beliefs),
        n_test,
        replace=False,
    )
    train_features = train.activations[train_indices]
    train_targets = train.beliefs[train_indices]
    test_features = test.activations[test_indices]
    test_targets = test.beliefs[test_indices]

    def fit_predict(permuted_targets: np.ndarray) -> np.ndarray:
        weight, bias = fit_affine_probe(
            train_features,
            permuted_targets,
            ridge=PROBE_RIDGE,
        )
        return probe_predict(weight, bias, test_features)

    null = held_out_permutation_null(
        train_targets,
        fit_predict,
        test_targets,
        n_permutations=n_permutations,
        seed=permutation_seed,
    )
    real_subset_mse = mean_squared_error(
        fit_predict(train_targets),
        test_targets,
    )
    quantiles = np.quantile(null, [0.05, 0.5, 0.95])
    return {
        "permutation_real_mse": real_subset_mse,
        "permutation_null_mse_p05": float(quantiles[0]),
        "permutation_null_mse_p50": float(quantiles[1]),
        "permutation_null_mse_p95": float(quantiles[2]),
        "permutation_null_p_value_lower_tail": float(
            (1 + np.count_nonzero(null <= real_subset_mse)) / (len(null) + 1)
        ),
        "permutation_null_n": int(n_permutations),
        "permutation_null_n_fit": int(n_train),
        "permutation_null_n_test": int(n_test),
    }


def _draw_probe_pair(
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
        title=f"{title}: exact Bayesian target",
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
            f"global ratio={result.metrics['global_mse_ratio']:.3f}"
        ),
        labels=("s0", "s1", "s2"),
    )


def plot_probe_pair(
    result: ProbeResult,
    *,
    title: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    _draw_probe_pair(axes, result, title=title)
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
    """Compare exact and decoded simplexes before and after training."""

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), squeeze=False)
    _draw_probe_pair(axes[0], initial, title=f"{condition} — init")
    _draw_probe_pair(axes[1], final, title=f"{condition} — final")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_probe_trajectory(
    checkpoints: list[dict[str, Any]],
    *,
    condition: str,
    path: Path,
) -> None:
    """Plot normalized probe MSE and task success for every checkpoint."""

    steps = np.asarray(
        [point["agent_steps"] for point in checkpoints],
        dtype=np.float64,
    )
    mse_ratio = np.asarray(
        [point["global_mse_ratio"] for point in checkpoints],
        dtype=np.float64,
    )
    success = 100.0 * np.asarray(
        [point["token_accuracy_greedy"] for point in checkpoints],
        dtype=np.float64,
    )
    figure, left = plt.subplots(figsize=(8.2, 4.8))
    right = left.twinx()
    left.plot(
        steps,
        mse_ratio,
        marker="o",
        color="#355c9a",
        label="Held-out global MSE / target variance",
    )
    right.plot(
        steps,
        success,
        marker="s",
        color="#c45135",
        label="Greedy token success",
    )
    right.axhline(
        100.0 * BAYESIAN_OPTIMAL_ACCURACY,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label=(
            "Bayes optimal, stationary 10-token context "
            f"({100.0 * BAYESIAN_OPTIMAL_ACCURACY:.3f}%)"
        ),
    )
    left.set_xlabel("Environment steps")
    left.set_ylabel("Normalized probe MSE (lower is better)", color="#355c9a")
    right.set_ylabel("Task success (%)", color="#c45135")
    left.set_title(condition.replace("_", " "))
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="best",
        fontsize=8,
    )
    left.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    agent_steps: int | None = None,
    train_steps: int | None = None,
    test_steps: int | None = None,
) -> ProbeResult:
    """Fit and evaluate the README-standard probe on disjoint rollouts."""

    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    context.results_dir.mkdir(parents=True, exist_ok=True)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = train_steps or (4_096 if context.smoke else 60_000)
    test_steps = test_steps or (
        4_096 if context.smoke else FULL_TEST_STEPS
    )
    warmup = 4 if context.smoke else 64
    n_resamples = SMOKE_RESAMPLES if context.smoke else FULL_RESAMPLES
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
            initial_belief, outcome_operator, initial_operator = (
                make_transducer_target(environment)
            )
        finally:
            environment.close()
        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": _device(context),
            "warmup": warmup,
            "n_envs": N_ENVS,
            "initial_belief": initial_belief,
            "action_outcome_operator": outcome_operator,
            "initial_outcome_operator": initial_operator,
        }
        train = collect_probe_data(
            n_steps=train_steps,
            seed=streams["probe_train"],
            **common,
        )
        test = collect_probe_data(
            n_steps=test_steps,
            seed=streams["probe_test"],
            **common,
        )

    target_error = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "Bayesian target is misaligned with environment diagnostics: "
            f"{target_error:.3e}"
        )
    weight, bias = fit_affine_probe(
        train.activations,
        train.beliefs,
        ridge=PROBE_RIDGE,
    )
    predicted = probe_predict(weight, bias, test.activations)
    global_metrics = global_mse_metrics(predicted, test.beliefs)
    fine_metrics = conditional_mse_metrics(
        predicted,
        test.beliefs,
        branch_keys(test, depth=2),
        min_group_size=MIN_GROUP_SIZE,
    )
    clusters = _episode_clusters(test)
    bootstrap = cluster_bootstrap_statistics(
        clusters,
        lambda indices: mean_squared_error(
            predicted[indices],
            test.beliefs[indices],
        ),
        n_resamples=n_resamples,
        seed=seed_sequence_to_int(streams["bootstrap"], bits=32),
    )
    mse_ci_low, mse_ci_high = percentile_interval(bootstrap)
    permutation_metrics = _permutation_null_metrics(
        train,
        test,
        n_permutations=n_resamples,
        sample_seed=seed_sequence_to_int(
            streams["permutation_sample"],
            bits=32,
        ),
        permutation_seed=seed_sequence_to_int(
            streams["permutation"],
            bits=32,
        ),
    )
    r_squared = r2_score(predicted, test.beliefs)
    metrics = {
        "condition": condition,
        "checkpoint_step": agent_steps,
        "is_untrained": agent_steps == 0,
        "target": "exact_predictive_bayesian_belief",
        "probe": "held_out_affine_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "representation": "post_final_layer_norm",
        "sampling_distribution": "process_weighted_rollout",
        "policy_mode": "greedy",
        "warmup": warmup,
        "n_envs": N_ENVS,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "branch_depth": 2,
        "min_group_size": MIN_GROUP_SIZE,
        **global_metrics,
        **fine_metrics,
        "r_squared": r_squared,
        "mse_ci_95_low": mse_ci_low,
        "mse_ci_95_high": mse_ci_high,
        "bootstrap_n": n_resamples,
        "bootstrap_cluster": "environment_episode",
        **permutation_metrics,
        "token_accuracy_greedy": float(test.rewards.mean()),
        "bayesian_optimal_accuracy_context_10": BAYESIAN_OPTIMAL_ACCURACY,
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        "target_consistency_max_abs": target_error,
        "interpretation": (
            "Affine decodability does not establish causal policy use."
        ),
    }
    if agent_steps == 0:
        metrics["untrained_mse"] = metrics["mse"]
    sample_size = min(PLOT_SAMPLE_SIZE, len(test.beliefs))
    sample_rng = np.random.default_rng(streams["plot_sample"])
    indices = sample_rng.choice(len(test.beliefs), sample_size, replace=False)
    result = ProbeResult(
        metrics=metrics,
        target_display=test.beliefs[indices],
        decoded_display=_simplex_display(predicted[indices]),
    )
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    plot_probe_pair(
        result,
        title=condition.replace("_", " "),
        path=context.results_dir / "belief_simplex.png",
    )
    return result
