"""Held-out belief probes and factor-geometry diagnostics for Cassandra."""

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
from analysis.probes import (
    conditional_mse_metrics,
    fit_affine_probe,
    global_mse_metrics,
    probe_predict,
    r2_score,
)
from envs.cassandra_machine import N_COMPONENTS, N_CONDITIONS, N_OBSERVATIONS
from experiments.cassandra_belief_factoring_2026_08.probe import (
    CassandraProbeData,
    collect_probe_data,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences


PROBE_RIDGE = 1e-6
N_ENVS = 8
SMOKE_PROBE_STEPS = 2_048
FULL_TRAIN_STEPS = 60_000
FULL_TEST_STEPS = 80_000
_STREAM_KEYS = {
    "probe_train": (400,),
    "probe_test": (401,),
}
TARGET_DESCRIPTIONS = {
    "joint_belief": "Exact posterior over all 4^4 joint component states.",
    "component_contrast": (
        "Four labeled component marginals in three independent simplex "
        "coordinates each (12 coordinates)."
    ),
    "identity_deviation": (
        "Labeled component marginals after subtracting their component mean; "
        "this isolates identity-specific information beyond a coarse average."
    ),
    "aggregate_contrast": (
        "Mean component marginal in three independent simplex coordinates."
    ),
    "labeled_expected_condition": (
        "Expected condition of each named component in fixed component order."
    ),
    "sorted_expected_condition": (
        "Permutation-invariant sorted expected component conditions."
    ),
    "next_operate_pass_probability": (
        "Posterior probability that the next operate action emits pass."
    ),
    "expected_action_reward": (
        "Posterior expected immediate reward for every available action."
    ),
    "broken_count_distribution": (
        "Posterior distribution over the number of broken components."
    ),
    "total_correlation": (
        "KL divergence from the joint posterior to the product of marginals."
    ),
}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    metrics: dict[str, Any]


def _device(context: RunContext) -> str:
    """Collect probe rollouts on CPU so training retains the full GPU budget."""

    return "cpu"


def variance_geometry(
    values: np.ndarray,
    *,
    max_spectrum_entries: int = 32,
) -> dict[str, Any]:
    """Report PCA CEV thresholds and participation ratio."""

    centered = np.asarray(values, dtype=np.float64)
    if centered.ndim != 2 or len(centered) < 2:
        raise ValueError("PCA values must have shape (N, D) with N >= 2")
    centered = centered - centered.mean(axis=0)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2
    positive = spectrum[spectrum > np.finfo(np.float64).eps * spectrum.max()]
    total = float(spectrum.sum())
    if total <= 0.0 or not len(positive):
        return {
            "rank": 0,
            "cev90_dimension": 0,
            "cev95_dimension": 0,
            "cev99_dimension": 0,
            "participation_ratio": 0.0,
            "explained_variance_fraction": [],
            "cumulative_explained_variance": [],
        }
    fractions = spectrum / total
    cumulative = np.cumsum(fractions)

    def threshold_dimension(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction) + 1)

    count = min(max_spectrum_entries, len(fractions))
    return {
        "rank": int(len(positive)),
        "cev90_dimension": threshold_dimension(0.90),
        "cev95_dimension": threshold_dimension(0.95),
        "cev99_dimension": threshold_dimension(0.99),
        "participation_ratio": float(
            np.square(spectrum.sum()) / np.square(spectrum).sum()
        ),
        "explained_variance_fraction": fractions[:count].tolist(),
        "cumulative_explained_variance": cumulative[:count].tolist(),
    }


def readout_subspace(weight: np.ndarray, *, rank: int) -> np.ndarray:
    """Return dominant orthonormal feature directions of a readout."""

    left, _, _ = np.linalg.svd(
        np.asarray(weight, dtype=np.float64),
        full_matrices=False,
    )
    return left[:, : min(rank, left.shape[1])]


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return normalized principal-angle overlap in [0, 1]."""

    denominator = min(left.shape[1], right.shape[1])
    if denominator == 0:
        return float("nan")
    return float(np.square(left.T @ right).sum() / denominator)


def _whiten(target: np.ndarray) -> tuple[np.ndarray, int]:
    centered = np.asarray(target, dtype=np.float64)
    centered = centered - centered.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > max(float(eigenvalues.max()) * 1e-10, 1e-12)
    if not keep.any():
        return centered, 0
    transform = (
        eigenvectors[:, keep]
        / np.sqrt(eigenvalues[keep])[None, :]
    )
    return centered @ transform, int(keep.sum())


def factor_subspace_geometry(
    activations: np.ndarray,
    component_targets: np.ndarray,
) -> dict[str, Any]:
    """Measure whether component beliefs share or separate readout directions."""

    components = np.asarray(component_targets).reshape(
        len(component_targets),
        N_COMPONENTS,
        N_CONDITIONS - 1,
    )
    bases = []
    ranks = []
    for component in range(N_COMPONENTS):
        whitened, rank = _whiten(components[:, component, :])
        weight, _ = fit_affine_probe(
            activations,
            whitened,
            ridge=PROBE_RIDGE,
        )
        bases.append(readout_subspace(weight, rank=rank))
        ranks.append(rank)
    overlaps = {
        f"component_{left}_vs_{right}": subspace_overlap(
            bases[left],
            bases[right],
        )
        for left in range(N_COMPONENTS)
        for right in range(left + 1, N_COMPONENTS)
    }
    finite = [value for value in overlaps.values() if np.isfinite(value)]
    union = np.concatenate(bases, axis=1)
    return {
        "target_whitening": "per_component_full_covariance",
        "component_ranks": ranks,
        "pairwise_overlap": overlaps,
        "mean_pairwise_overlap": (
            float(np.mean(finite)) if finite else float("nan")
        ),
        "union_rank": int(np.linalg.matrix_rank(union, tol=1e-8)),
        "sum_component_ranks": int(sum(ranks)),
    }


def _observation_groups(observations: np.ndarray) -> np.ndarray:
    symbols = np.asarray(
        observations[:, :N_OBSERVATIONS]
    ).argmax(axis=1)
    action_features = np.asarray(observations[:, N_OBSERVATIONS:])
    action_count = action_features.shape[1]
    has_previous_action = action_features.sum(axis=1) > 0.5
    previous_actions = np.where(
        has_previous_action,
        action_features.argmax(axis=1) + 1,
        0,
    )
    return symbols * (action_count + 1) + previous_actions


def _fit_target(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    *,
    groups: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    weight, bias = fit_affine_probe(
        train_features,
        train_target,
        ridge=PROBE_RIDGE,
    )
    predicted = probe_predict(weight, bias, test_features)
    metrics = {
        **global_mse_metrics(predicted, test_target),
        "r_squared": r2_score(predicted, test_target),
        **conditional_mse_metrics(
            predicted,
            test_target,
            groups,
            min_group_size=10,
        ),
    }
    return metrics, predicted, weight


def _factor_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    predicted = predicted.reshape(
        len(predicted),
        N_COMPONENTS,
        N_CONDITIONS - 1,
    )
    target = target.reshape(
        len(target),
        N_COMPONENTS,
        N_CONDITIONS - 1,
    )
    return {
        f"component_{component}": {
            **global_mse_metrics(
                predicted[:, component, :],
                target[:, component, :],
            ),
            "r_squared": r2_score(
                predicted[:, component, :],
                target[:, component, :],
            ),
        }
        for component in range(N_COMPONENTS)
    }


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    agent_steps: int,
) -> ProbeResult:
    """Fit matched-history affine probes for one checkpoint."""

    if context.seed is None:
        raise ValueError("Cassandra probing requires a resolved seed")
    context.results_dir.mkdir(parents=True, exist_ok=True)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_TRAIN_STEPS
    test_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_TEST_STEPS
    warmup = 16 if context.smoke else 64

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        env_config["diagnostics"] = True
        action_scope = str(env_config.get("action_scope", "global"))

        def make_environment():
            return env_class(env_config)

        common = {
            "module": module,
            "env_factory": make_environment,
            "n_envs": N_ENVS,
            "device": _device(context),
            "warmup": warmup,
            "action_scope": action_scope,
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

    consistency = max(
        train.marginal_consistency_max_abs,
        test.marginal_consistency_max_abs,
    )
    if consistency > 1e-10:
        raise AssertionError(
            "Cassandra factored diagnostics disagree with the exact joint "
            f"belief: {consistency:.3e}"
        )

    groups = _observation_groups(test.observations)
    targets: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for name in TARGET_DESCRIPTIONS:
        metrics, predicted, weight = _fit_target(
            train.activations,
            test.activations,
            train.targets[name],
            test.targets[name],
            groups=groups,
        )
        observation_metrics, _, _ = _fit_target(
            train.observations,
            test.observations,
            train.targets[name],
            test.targets[name],
            groups=groups,
        )
        metrics["observation_only_r_squared"] = observation_metrics["r_squared"]
        metrics["representation_gain_over_observation_r2"] = (
            metrics["r_squared"] - observation_metrics["r_squared"]
        )
        targets[name] = metrics
        predictions[name] = predicted
        weights[name] = weight

    factor_metrics = _factor_metrics(
        predictions["component_contrast"],
        test.targets["component_contrast"],
    )
    factor_geometry = factor_subspace_geometry(
        train.activations,
        train.targets["component_contrast"],
    )
    activation_geometry = variance_geometry(test.activations)
    target_geometry = {
        name: variance_geometry(values)
        for name, values in test.targets.items()
    }
    aggregate_r2 = float(targets["aggregate_contrast"]["r_squared"])
    identity_r2 = float(targets["identity_deviation"]["r_squared"])
    labeled_r2 = float(targets["labeled_expected_condition"]["r_squared"])
    sorted_r2 = float(targets["sorted_expected_condition"]["r_squared"])
    hypothesis = {
        "aggregate_r_squared": aggregate_r2,
        "identity_deviation_r_squared": identity_r2,
        "coarse_over_identity_r2_advantage": aggregate_r2 - identity_r2,
        "sorted_expected_condition_r_squared": sorted_r2,
        "labeled_expected_condition_r_squared": labeled_r2,
        "permutation_invariant_r2_advantage": sorted_r2 - labeled_r2,
        "mean_component_subspace_overlap": factor_geometry[
            "mean_pairwise_overlap"
        ],
        "interpretation": (
            "A positive coarse-over-identity gap, positive sorted-over-labeled "
            "gap, and high component-subspace overlap jointly support a coarse "
            "permutation-invariant representation. No one metric is decisive."
        ),
    }
    action_counts = np.bincount(
        test.actions.reshape(-1),
        minlength=test.observations.shape[1] - N_OBSERVATIONS,
    )
    metrics = {
        "condition": condition,
        "checkpoint_step": agent_steps,
        "is_untrained": agent_steps == 0,
        "probe": "held_out_affine_ridge_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "representation": "pre_final_layer_norm_decision_token",
        "sampling_distribution": "fixed_checkpoint_independent_behavior_policy",
        "policy_input": (
            "current_symbol_one_hot_plus_previous_action_one_hot"
        ),
        "action_scope": action_scope,
        "reward_in_policy_input": False,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "n_envs": N_ENVS,
        "warmup": warmup,
        "target_descriptions": TARGET_DESCRIPTIONS,
        "targets": targets,
        "factor_specific": factor_metrics,
        "geometry": {
            "activation_pca": activation_geometry,
            "target_pca": target_geometry,
            "component_readout_subspaces": factor_geometry,
        },
        "hypothesis_diagnostics": hypothesis,
        "behavior_action_fractions": (
            action_counts / max(action_counts.sum(), 1)
        ).tolist(),
        "behavior_reward_mean": float(test.rewards.mean()),
        "n_fit": len(train.activations),
        "n_test": len(test.activations),
        "target_consistency_max_abs": consistency,
        "caveat": (
            "Linear decodability and PCA geometry do not establish causal "
            "policy use; an addressable-action intervention is the causal "
            "follow-up."
        ),
    }
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    plot_cev(metrics, path=context.results_dir / "cev_curves.png")
    return ProbeResult(metrics=metrics)


def plot_cev(metrics: dict[str, Any], *, path: Path) -> None:
    """Plot activation and primary-target cumulative explained variance."""

    geometry = metrics["geometry"]
    curves = {
        "Transformer activation": geometry["activation_pca"],
        "Labeled component belief": geometry["target_pca"][
            "component_contrast"
        ],
        "Identity residual": geometry["target_pca"]["identity_deviation"],
        "Mean aggregate belief": geometry["target_pca"][
            "aggregate_contrast"
        ],
    }
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    for label, values in curves.items():
        cumulative = values["cumulative_explained_variance"]
        axis.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            marker="." if len(cumulative) < 16 else None,
            label=label,
        )
    axis.axhline(0.95, color="#333333", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Principal components")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0.0, 1.01)
    axis.set_title(
        f"Cassandra belief geometry — step {metrics['checkpoint_step']}"
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_probe_trajectory(
    checkpoints: list[dict[str, Any]],
    *,
    path: Path,
) -> None:
    """Plot coarse and identity-sensitive decodability across training."""

    steps = np.asarray([point["agent_steps"] for point in checkpoints])
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for name, label in (
        ("aggregate_contrast", "Mean aggregate belief"),
        ("identity_deviation", "Identity-only residual"),
        ("sorted_expected_condition", "Sorted component health"),
        ("labeled_expected_condition", "Labeled component health"),
    ):
        axes[0].plot(
            steps,
            [
                point["targets"][name]["r_squared"]
                for point in checkpoints
            ],
            marker="o",
            label=label,
        )
    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Held-out affine-probe R²")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        steps,
        [
            point["hypothesis_diagnostics"][
                "coarse_over_identity_r2_advantage"
            ]
            for point in checkpoints
        ],
        marker="o",
        label="Aggregate R² − identity residual R²",
    )
    axes[1].plot(
        steps,
        [
            point["hypothesis_diagnostics"][
                "mean_component_subspace_overlap"
            ]
            for point in checkpoints
        ],
        marker="s",
        label="Mean component subspace overlap",
    )
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Coarseness diagnostic")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Environment steps (0 = initialization)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
