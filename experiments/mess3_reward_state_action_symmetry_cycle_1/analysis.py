"""README-compliant affine belief probes for the action-symmetry study."""

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
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    branch_keys,
    collect_probe_data,
    make_transducer_target,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int


PROBE_RIDGE = 1e-6
MIN_GROUP_SIZE = 50
N_ENVS = 16
FULL_RESAMPLES = 1_000
SMOKE_RESAMPLES = 100
PERMUTATION_SAMPLE_CAP = 4_096
_STREAM_KEYS = {
    "probe_train": (300,),
    "probe_test": (301,),
    "bootstrap": (302,),
    "permutation": (303,),
    "permutation_sample": (304,),
    "plot_sample": (305,),
}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    metrics: dict[str, Any]
    targets: np.ndarray
    predictions: np.ndarray


def _device(context: RunContext) -> str:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _episode_clusters(data: ProbeData) -> np.ndarray:
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


def _permutation_metrics(
    train: ProbeData,
    test: ProbeData,
    *,
    n_permutations: int,
    sample_seed: int,
    permutation_seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(sample_seed)
    train_indices = rng.choice(
        len(train.beliefs),
        min(PERMUTATION_SAMPLE_CAP, len(train.beliefs)),
        replace=False,
    )
    test_indices = rng.choice(
        len(test.beliefs),
        min(PERMUTATION_SAMPLE_CAP, len(test.beliefs)),
        replace=False,
    )
    train_features = train.activations[train_indices]
    train_targets = train.beliefs[train_indices]
    test_features = test.activations[test_indices]
    test_targets = test.beliefs[test_indices]

    def fit_predict(targets: np.ndarray) -> np.ndarray:
        weight, bias = fit_affine_probe(
            train_features,
            targets,
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
    real_mse = mean_squared_error(fit_predict(train_targets), test_targets)
    quantiles = np.quantile(null, [0.05, 0.5, 0.95])
    return {
        "permutation_real_mse": real_mse,
        "permutation_null_mse_p05": float(quantiles[0]),
        "permutation_null_mse_p50": float(quantiles[1]),
        "permutation_null_mse_p95": float(quantiles[2]),
        "permutation_null_p_value_lower_tail": float(
            (1 + np.count_nonzero(null <= real_mse)) / (len(null) + 1)
        ),
        "permutation_null_n": int(n_permutations),
    }


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    agent_steps: int,
) -> ProbeResult:
    """Fit on one rollout stream and score on an independent stream."""

    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = 4_096 if context.smoke else 60_000
    test_steps = 4_096 if context.smoke else 80_000
    warmup = 4 if context.smoke else 64
    n_resamples = SMOKE_RESAMPLES if context.smoke else FULL_RESAMPLES

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        env_config["diagnostics"] = {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        }

        def make_environment():
            return env_class(env_config)

        environment = make_environment()
        try:
            transducer_target = make_transducer_target(environment)
        finally:
            environment.close()
        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": _device(context),
            "warmup": warmup,
            "n_envs": N_ENVS,
            "initial_belief": transducer_target[0],
            "action_outcome_operator": transducer_target[1],
            "initial_outcome_operator": transducer_target[2],
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
            "transducer target is misaligned with environment diagnostics: "
            f"{target_error:.3e}"
        )
    weight, bias = fit_affine_probe(
        train.activations,
        train.beliefs,
        ridge=PROBE_RIDGE,
    )
    predicted = probe_predict(weight, bias, test.activations)
    bootstrap = cluster_bootstrap_statistics(
        _episode_clusters(test),
        lambda indices: mean_squared_error(
            predicted[indices],
            test.beliefs[indices],
        ),
        n_resamples=n_resamples,
        seed=seed_sequence_to_int(streams["bootstrap"], bits=32),
    )
    mse_ci_low, mse_ci_high = percentile_interval(bootstrap)
    discrete_actions = np.asarray(test.actions, dtype=np.int64).reshape(-1)
    action_counts = np.bincount(discrete_actions, minlength=3)
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
        **global_mse_metrics(predicted, test.beliefs),
        **conditional_mse_metrics(
            predicted,
            test.beliefs,
            branch_keys(test, depth=2),
            min_group_size=MIN_GROUP_SIZE,
        ),
        "r_squared": r2_score(predicted, test.beliefs),
        "mse_ci_95_low": mse_ci_low,
        "mse_ci_95_high": mse_ci_high,
        "bootstrap_n": n_resamples,
        "bootstrap_cluster": "environment_episode",
        **_permutation_metrics(
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
        ),
        "reward_state_2_fraction_greedy": float(test.rewards.mean()),
        "greedy_action_fractions": (
            action_counts / max(action_counts.sum(), 1)
        ).tolist(),
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        "target_consistency_max_abs": target_error,
        "interpretation": (
            "Affine decodability does not establish causal policy use."
        ),
    }
    if agent_steps == 0:
        metrics["untrained_mse"] = metrics["mse"]
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    sample_size = min(20_000, len(test.beliefs))
    rng = np.random.default_rng(streams["plot_sample"])
    indices = rng.choice(len(test.beliefs), sample_size, replace=False)
    return ProbeResult(
        metrics=metrics,
        targets=test.beliefs[indices],
        predictions=predicted[indices],
    )


def plot_probe(result: ProbeResult, *, title: str, path: Path) -> None:
    """Plot exact and affine-decoded beliefs without altering scored values."""

    display = np.clip(result.predictions, 0.0, None)
    display /= np.maximum(display.sum(axis=1, keepdims=True), 1e-12)
    colors = np.clip(result.targets, 0.0, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    simplex_scatter(
        axes[0],
        result.targets,
        colors=colors,
        s=0.3,
        alpha=0.2,
        title=f"{title}: exact belief",
        labels=("s0", "s1", "s2"),
    )
    simplex_scatter(
        axes[1],
        display,
        colors=colors,
        s=0.3,
        alpha=0.2,
        title=(
            f"{title}: affine decoded\n"
            f"MSE={result.metrics['mse']:.5f}"
        ),
        labels=("s0", "s1", "s2"),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
