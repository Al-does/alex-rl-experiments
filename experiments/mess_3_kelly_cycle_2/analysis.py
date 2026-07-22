"""Aligned belief probes and wager diagnostics for Kelly cycle 2."""

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
from experiments.mess3_token_guess_cycle_1.analysis import (
    fit_reduced_rank_affine,
)
from experiments.mess_3_kelly_cycle_1.kelly import (
    COLLAPSE_THRESHOLD,
    expected_log_growth,
    kelly_fraction,
    realized_log_growth,
)
from experiments.mess_3_kelly_cycle_1.task import RawNextTokenTask
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences


PROBE_RANK = 2
_STREAM_KEYS = {
    "probe_train": (200,),
    "probe_test": (201,),
    "plot_sample": (202,),
}


@dataclass(frozen=True, slots=True)
class Cycle2ProbeResult:
    metrics: dict[str, Any]
    target_display: np.ndarray
    decoded_display: np.ndarray


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


def _plot_beliefs(
    target: np.ndarray,
    decoded: np.ndarray,
    *,
    condition: str,
    r_squared: float,
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
        title=(
            f"{condition.replace('_', ' ')}: rank-2 decoded hidden state\n"
            f"held-out R²={r_squared:.4f}"
        ),
        labels=("s0", "s1", "s2"),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_wagers(
    oracle: np.ndarray,
    actual: np.ndarray,
    *,
    condition: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.hexbin(
        oracle,
        actual,
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0)
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Bayes-optimal wager for selected token",
        ylabel="Policy wager",
        title=condition.replace("_", " "),
    )
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


@torch.no_grad()
def _policy_outputs(
    module: Any,
    activations: np.ndarray,
    actions: np.ndarray,
    *,
    condition: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    embeddings = torch.as_tensor(
        activations,
        dtype=torch.float32,
        device=device,
    )
    logits = module.action_distribution_inputs(embeddings)
    probabilities = torch.softmax(logits, dim=-1)
    action_tensor = torch.as_tensor(
        actions,
        dtype=torch.long,
        device=device,
    )
    selected_probability = probabilities.gather(
        -1,
        action_tensor.unsqueeze(-1),
    ).squeeze(-1)
    if not hasattr(module, "wager_fraction"):
        wager = None
    elif condition.startswith("conditional_decoupled_kelly_"):
        fractions = module.wager_fraction(embeddings)
        wager = fractions.gather(
            -1,
            action_tensor.unsqueeze(-1),
        ).squeeze(-1)
    else:
        wager = module.wager_fraction(embeddings)
    return (
        selected_probability.cpu().numpy(),
        None if wager is None else wager.cpu().numpy(),
    )


def probe_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    train_steps: int | None = None,
    test_steps: int | None = None,
) -> Cycle2ProbeResult:
    """Evaluate full belief geometry and action/wager quality."""

    if context.seed is None:
        raise ValueError("cycle-2 probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    if train_steps is None:
        train_steps = 512 if context.smoke else 60_000
    if test_steps is None:
        test_steps = 256 if context.smoke else 30_000
    warmup = 4 if context.smoke else 64
    device = _device(context)

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        environment_class = algorithm.config.env
        environment_config = dict(algorithm.config.env_config)
        environment_config["task"] = {
            "class": (
                "experiments.mess_3_kelly_cycle_1.task:"
                f"{RawNextTokenTask.__name__}"
            )
        }
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
            emission_matrix = environment.model.emission_matrix.copy()
        finally:
            environment.close()

        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "device": device,
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
        actions = np.asarray(test.actions, dtype=np.int64).reshape(-1)
        policy_probability, wager = _policy_outputs(
            module,
            test.activations,
            actions,
            condition=condition,
            device=device,
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

    exact_probabilities = test.beliefs @ emission_matrix
    exact_selected_probability = exact_probabilities[
        np.arange(len(actions)),
        actions,
    ]
    oracle_selected_wager = kelly_fraction(exact_selected_probability)
    oracle_selected_growth = expected_log_growth(
        exact_selected_probability,
        oracle_selected_wager,
    )
    best_probability = exact_probabilities.max(axis=-1)
    best_wager = kelly_fraction(best_probability)
    oracle_ceiling = expected_log_growth(best_probability, best_wager)
    correct = np.asarray(test.rewards, dtype=np.float64)

    metrics: dict[str, Any] = {
        "condition": condition,
        "target": "exact_predictive_bayesian_belief",
        "probe": "held_out_reduced_rank_affine_least_squares",
        "probe_rank": PROBE_RANK,
        "r_squared": r_squared,
        "mse": float(np.square(predicted - test.beliefs).mean()),
        "token_accuracy_greedy": float(correct.mean()),
        "policy_selected_probability_mean": float(policy_probability.mean()),
        "oracle_selected_action_log_growth_mean": float(
            oracle_selected_growth.mean()
        ),
        "oracle_policy_ceiling_log_growth_mean": float(oracle_ceiling.mean()),
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        "target_consistency_max_abs": target_error,
    }
    if wager is None:
        metrics.update(
            {
                "wager_mean": None,
                "wager_collapse_fraction": None,
                "wager_vs_oracle_rmse": None,
                "realized_log_growth_mean": None,
                "expected_log_growth_mean": None,
                "expected_growth_vs_ceiling_ratio": None,
            }
        )
    else:
        realized_growth = realized_log_growth(correct > 0.5, wager)
        expected_growth = expected_log_growth(
            exact_selected_probability,
            wager,
        )
        ceiling_mean = float(oracle_ceiling.mean())
        metrics.update(
            {
                "wager_mean": float(wager.mean()),
                "wager_collapse_fraction": float(
                    (wager < COLLAPSE_THRESHOLD).mean()
                ),
                "wager_vs_oracle_rmse": float(
                    np.sqrt(np.square(wager - oracle_selected_wager).mean())
                ),
                "realized_log_growth_mean": float(realized_growth.mean()),
                "expected_log_growth_mean": float(expected_growth.mean()),
                "expected_growth_vs_ceiling_ratio": (
                    float(expected_growth.mean()) / ceiling_mean
                    if ceiling_mean > 0.0
                    else None
                ),
            }
        )

    sample_size = min(20_000, len(test.beliefs))
    sample_rng = np.random.default_rng(streams["plot_sample"])
    indices = sample_rng.choice(len(test.beliefs), sample_size, replace=False)
    target_display = test.beliefs[indices]
    decoded_display = _simplex_display(predicted[indices])
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    _plot_beliefs(
        target_display,
        decoded_display,
        condition=condition,
        r_squared=r_squared,
        path=context.results_dir / "belief_simplex.png",
    )
    if wager is not None:
        _plot_wagers(
            oracle_selected_wager[indices],
            wager[indices],
            condition=condition,
            path=context.results_dir / "wager_calibration.png",
        )
    return Cycle2ProbeResult(metrics, target_display, decoded_display)
