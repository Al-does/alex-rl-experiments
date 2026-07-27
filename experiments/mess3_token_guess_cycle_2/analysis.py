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
from analysis.probes import r2_score
from envs.mess3.model import passive_model
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences


PROBE_RANK = 2
CONTEXT_LENGTH = 10
_STREAM_KEYS = {
    "probe_train": (200,),
    "probe_test": (201,),
    "plot_sample": (202,),
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
        s=0.6,
        alpha=0.35,
        title=f"{title}: exact Bayesian target",
        labels=("s0", "s1", "s2"),
    )
    simplex_scatter(
        axes[1],
        result.decoded_display,
        colors=colors,
        s=0.6,
        alpha=0.35,
        title=(
            f"{title}: rank-2 decoded residual stream\n"
            f"held-out R²={result.metrics['r_squared']:.4f}"
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
    """Plot probe R² and held-out task success for every checkpoint."""

    steps = np.asarray(
        [point["agent_steps"] for point in checkpoints],
        dtype=np.float64,
    )
    r_squared = np.asarray(
        [point["r_squared"] for point in checkpoints],
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
        r_squared,
        marker="o",
        color="#355c9a",
        label="Held-out belief probe R²",
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
    left.set_ylabel("Held-out probe R²", color="#355c9a")
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
    train_steps: int | None = None,
    test_steps: int | None = None,
) -> ProbeResult:
    """Fit and evaluate a rank-2 belief probe on disjoint rollout seeds."""

    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    context.results_dir.mkdir(parents=True, exist_ok=True)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = train_steps or (512 if context.smoke else 60_000)
    test_steps = test_steps or (256 if context.smoke else 30_000)
    warmup = 4 if context.smoke else 64
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
    weight, bias = fit_reduced_rank_affine(
        train.activations,
        train.beliefs,
        rank=PROBE_RANK,
    )
    predicted = test.activations @ weight + bias
    r_squared = r2_score(predicted, test.beliefs)
    metrics = {
        "condition": condition,
        "target": "exact_predictive_bayesian_belief",
        "probe": "held_out_reduced_rank_affine_least_squares",
        "probe_rank": PROBE_RANK,
        "r_squared": r_squared,
        "mse": float(np.square(predicted - test.beliefs).mean()),
        "token_accuracy_greedy": float(test.rewards.mean()),
        "bayesian_optimal_accuracy_context_10": BAYESIAN_OPTIMAL_ACCURACY,
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        "target_consistency_max_abs": target_error,
    }
    sample_size = min(20_000, len(test.beliefs))
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
