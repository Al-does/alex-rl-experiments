"""Measure what the belief probe reads out of a network that never trained.

Every other arm in this study is scored by a rank-2 affine probe on the
transformer's hidden state. A randomly initialised causal transformer over
one-hot tokens is already a nonlinear filter of the recent token window, and
the MESS3 predictive belief is a smooth function of exactly that window, so the
probe can succeed without any learning at all. Without this floor a reported
R² cannot be attributed to the training objective.

The arm is deliberately identical to the trained conditions in environment,
architecture, probe, and rollout seeds. Only the weights are untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_token_guess_cycle_1.analysis import (
    PROBE_RANK,
    fit_reduced_rank_affine,
)
from experiments.mess3_token_guess_cycle_1.baselines import calibrate
from experiments.mess3_token_guess_cycle_1.comparison.experiment import (
    BASE_MODEL_CONFIG,
    ENV_CONFIG,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.seeding import named_seed_sequences
from learners.models.transformer import TransformerModel


INITIALISATIONS = (0, 1, 2)
_STREAM_KEYS = {"probe_train": (100,), "probe_test": (101,)}


def _probe_environment_config() -> dict[str, Any]:
    return {
        **ENV_CONFIG,
        "diagnostics": {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        },
    }


def probe_untrained_module(
    context: RunContext,
    *,
    initialisation: int,
) -> dict[str, Any]:
    """Score one freshly initialised transformer with the study's own probe."""

    if context.seed is None:
        raise ValueError("the untrained reference requires a resolved seed")
    config = _probe_environment_config()

    def make_environment():
        return HMMEnv(config)

    environment = make_environment()
    try:
        initial_belief, outcome_operator, initial_operator = (
            make_transducer_target(environment)
        )
        emission_matrix = environment.model.emission_matrix.copy()
        observation_space = environment.observation_space
        action_space = environment.action_space
    finally:
        environment.close()

    torch.manual_seed(context.seed + initialisation)
    module = RLModuleSpec(
        module_class=TransformerModel,
        model_config=dict(BASE_MODEL_CONFIG),
        observation_space=observation_space,
        action_space=action_space,
    ).build()

    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    common = {
        "module": module,
        "env_factory": make_environment,
        "policy_mode": "greedy",
        "device": "cpu",
        "warmup": 4 if context.smoke else 64,
        "initial_belief": initial_belief,
        "action_outcome_operator": outcome_operator,
        "initial_outcome_operator": initial_operator,
    }
    train = collect_probe_data(
        n_steps=512 if context.smoke else 60_000,
        seed=streams["probe_train"],
        **common,
    )
    test = collect_probe_data(
        n_steps=256 if context.smoke else 30_000,
        seed=streams["probe_test"],
        **common,
    )

    weight, bias = fit_reduced_rank_affine(
        train.activations,
        train.beliefs,
        rank=PROBE_RANK,
    )
    predicted = test.activations @ weight + bias
    residual = float(np.square(predicted - test.beliefs).sum())
    total = float(np.square(test.beliefs - test.beliefs.mean(axis=0)).sum())
    return {
        "initialisation": initialisation,
        "r_squared": 1.0 - residual / total,
        "token_accuracy_greedy": float(test.rewards.mean()),
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        **calibrate(
            train,
            test,
            predicted,
            emission_matrix=emission_matrix,
            fit=fit_reduced_rank_affine,
            rank=PROBE_RANK,
        ),
    }


def _findings(summary: dict[str, Any]) -> str:
    r_squared = summary["r_squared_mean"]
    lines = [
        "# Untrained-network reference",
        "",
        "A randomly initialised transformer, never optimised, scored by the "
        "same rank-2 affine belief probe as every trained arm.",
        "",
        "| init | belief R² | greedy accuracy | within-branch R² (depth 2) |",
        "|---:|---:|---:|---:|",
    ]
    for run in summary["initialisations"]:
        lines.append(
            f"| {run['initialisation']} | {run['r_squared']:.4f} | "
            f"{run['token_accuracy_greedy']:.4f} | "
            f"{run['r_squared_within_branch_depth2']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Mean untrained belief R²: {r_squared:.4f}. Any trained arm at or "
            "below this value has not been shown to represent the belief "
            "simplex; the probe is reading the recent token window that a "
            "random causal filter already exposes.",
            "",
        ]
    )
    return "\n".join(lines)


def run(context: RunContext):
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    initialisations = (0,) if context.smoke else INITIALISATIONS
    runs = [
        probe_untrained_module(context, initialisation=initialisation)
        for initialisation in initialisations
    ]
    scores = np.array([run["r_squared"] for run in runs])
    accuracies = np.array([run["token_accuracy_greedy"] for run in runs])
    summary = {
        "condition": "untrained_reference",
        "seed": context.seed,
        "smoke": context.smoke,
        "environment": ENV_CONFIG,
        "model": BASE_MODEL_CONFIG,
        "initialisations": runs,
        "r_squared_mean": float(scores.mean()),
        "r_squared_std": float(scores.std(ddof=0)),
        "token_accuracy_mean": float(accuracies.mean()),
    }
    outputs.write_json("untrained_reference_summary.json", summary)
    (context.results_dir / "findings.md").write_text(_findings(summary))
    return summary
