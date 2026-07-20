"""Held-out reduced-rank belief probes and simplex plots."""

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
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences


PROBE_RANK = 2
_STREAM_KEYS = {
    "probe_train": (100,),
    "probe_test": (101,),
    "plot_sample": (102,),
}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    metrics: dict[str, Any]
    target_display: np.ndarray
    decoded_display: np.ndarray


def fit_reduced_rank_affine(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the least-squares affine map constrained to a given output rank."""

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
    if (
        profile.learner_device == "mps"
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _simplex_display(points: np.ndarray) -> np.ndarray:
    clipped = np.clip(points, 0.0, None)
    return clipped / np.maximum(clipped.sum(axis=1, keepdims=True), 1e-12)


def _plot_pair(
    target: np.ndarray,
    decoded: np.ndarray,
    *,
    title: str,
    r2: float,
    path: Path,
) -> None:
    colors = np.clip(target, 0.0, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    simplex_scatter(
        axes[0],
        target,
        colors=colors,
        s=0.6,
        alpha=0.35,
        title="Exact Bayesian belief",
        labels=("s0", "s1", "s2"),
    )
    simplex_scatter(
        axes[1],
        decoded,
        colors=colors,
        s=0.6,
        alpha=0.35,
        title=f"{title}: rank-2 decoded hidden state\nheld-out R²={r2:.4f}",
        labels=("s0", "s1", "s2"),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
) -> ProbeResult:
    """Probe one checkpoint on independent train/test rollouts."""

    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = 512 if context.smoke else 60_000
    test_steps = 256 if context.smoke else 30_000
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
    r2 = r2_score(predicted, test.beliefs)
    metrics = {
        "condition": condition,
        "target": "exact_predictive_bayesian_belief",
        "probe": "held_out_reduced_rank_affine_least_squares",
        "probe_rank": PROBE_RANK,
        "r_squared": r2,
        "mse": float(np.square(predicted - test.beliefs).mean()),
        "token_accuracy_greedy": float(test.rewards.mean()),
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        "target_consistency_max_abs": target_error,
    }
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    sample_size = min(20_000, len(test.beliefs))
    sample_rng = np.random.default_rng(streams["plot_sample"])
    indices = sample_rng.choice(len(test.beliefs), sample_size, replace=False)
    target_display = test.beliefs[indices]
    decoded_display = _simplex_display(predicted[indices])
    _plot_pair(
        target_display,
        decoded_display,
        title=condition.replace("_", " "),
        r2=r2,
        path=context.results_dir / "belief_simplex.png",
    )
    return ProbeResult(metrics, target_display, decoded_display)


def plot_comparison(
    results: dict[str, ProbeResult],
    *,
    path: Path,
) -> None:
    """Plot exact and decoded belief geometry for all controlled conditions."""

    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(9.2, 4.0 * len(results)),
        squeeze=False,
    )
    for row, (condition, result) in enumerate(results.items()):
        colors = np.clip(result.target_display, 0.0, 1.0)
        simplex_scatter(
            axes[row, 0],
            result.target_display,
            colors=colors,
            s=0.5,
            alpha=0.32,
            title=f"{condition.replace('_', ' ')}: exact Bayesian",
            labels=("s0", "s1", "s2"),
        )
        simplex_scatter(
            axes[row, 1],
            result.decoded_display,
            colors=colors,
            s=0.5,
            alpha=0.32,
            title=(
                "rank-2 affine decode of hidden state\n"
                f"held-out R²={result.metrics['r_squared']:.4f}"
            ),
            labels=("s0", "s1", "s2"),
        )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)
