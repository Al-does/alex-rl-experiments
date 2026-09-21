"""Reference bands that make token-guess outcomes interpretable.

Both headline axes of this study have floors far above zero. On passive MESS3
with ``alpha=0.85`` a policy that only echoes the last visible token already
reaches 0.671 accuracy against a Bayes ceiling of 0.688, and a rank-2 affine
probe reads 0.80-0.92 belief R² out of features that carry nothing but the last
one or two tokens. A bare 0.855 or 0.6733 therefore cannot be read as evidence
that an agent represents the belief simplex.

These helpers score the trivial references on the very rollout the agent is
scored on, so both axes can be reported as a fraction of the range they are
able to move through.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.probes import conditional_residual_r2
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    branch_keys,
)


def _actions(data: ProbeData) -> np.ndarray:
    return np.asarray(data.actions, dtype=np.int64).reshape(len(data.beliefs))


def _one_hot(indices: np.ndarray, size: int) -> np.ndarray:
    encoded = np.zeros((len(indices), size + 1))
    encoded[np.arange(len(indices)), np.where(indices < 0, size, indices)] = 1.0
    return encoded[:, :size]


def expected_accuracy_band(
    data: ProbeData,
    emission_matrix: np.ndarray,
) -> dict[str, float]:
    """Score the agent, the echo baseline, and Bayes on one belief trajectory.

    Each policy is scored by its expected hit rate ``P(x_t = a_t | x_<t)``
    rather than by realized hits. The three numbers then share a single
    realization of the process, which removes the sampling noise that otherwise
    swamps an accuracy range only 0.017 wide.
    """

    emission = np.asarray(emission_matrix, dtype=np.float64)
    predictive = np.asarray(data.beliefs, dtype=np.float64) @ emission
    rows = np.arange(len(predictive))

    visible = np.asarray(data.tokens, dtype=np.int64)
    if (visible < 0).any():
        raise ValueError(
            "the echo baseline needs a visible token at every decision; "
            "raise the probe warmup above the token delay"
        )

    agent = float(predictive[rows, _actions(data)].mean())
    echo = float(predictive[rows, visible].mean())
    bayes = float(predictive.max(axis=1).mean())
    span = bayes - echo
    return {
        "expected_accuracy_agent": agent,
        "expected_accuracy_echo_last_token": echo,
        "expected_accuracy_bayes": bayes,
        "accuracy_range": span,
        "accuracy_fraction_of_range": (
            float("nan") if span <= 0.0 else (agent - echo) / span
        ),
    }


def trivial_feature_r2(
    train: ProbeData,
    test: ProbeData,
    *,
    n_tokens: int,
    fit,
    rank: int,
) -> dict[str, float]:
    """Fit the study's own probe to features that require no belief state.

    ``fit`` is the reduced-rank affine fitter used on hidden states, so these
    numbers are directly comparable to the reported ``r_squared``.
    """

    def score(train_features: np.ndarray, test_features: np.ndarray) -> float:
        weight, bias = fit(train_features, train.beliefs, rank=rank)
        predicted = test_features @ weight + bias
        residual = float(np.square(predicted - test.beliefs).sum())
        total = float(
            np.square(test.beliefs - test.beliefs.mean(axis=0)).sum()
        )
        return float("nan") if total == 0.0 else 1.0 - residual / total

    def window(data: ProbeData, depth: int) -> np.ndarray:
        columns = [_one_hot(data.tokens, n_tokens)]
        if depth == 2:
            columns.append(_one_hot(data.previous_tokens, n_tokens))
        return np.concatenate(columns, axis=1)

    return {
        "r_squared_last_token": score(window(train, 1), window(test, 1)),
        "r_squared_last_two_tokens": score(window(train, 2), window(test, 2)),
        "r_squared_own_action_only": score(
            _one_hot(_actions(train), n_tokens),
            _one_hot(_actions(test), n_tokens),
        ),
    }


def residual_r2_beyond_recent_tokens(
    predicted: np.ndarray,
    test: ProbeData,
) -> dict[str, float]:
    """Report probe accuracy after the recent-token branch means are removed."""

    return {
        f"r_squared_within_branch_depth{depth}": conditional_residual_r2(
            predicted,
            test.beliefs,
            branch_keys(test, depth),
            min_group_size=50,
        )
        for depth in (1, 2)
    }


def calibrate(
    train: ProbeData,
    test: ProbeData,
    predicted: np.ndarray,
    *,
    emission_matrix: np.ndarray,
    fit,
    rank: int,
) -> dict[str, Any]:
    """Bundle every reference the two headline numbers need to be readable."""

    return {
        **expected_accuracy_band(test, emission_matrix),
        **trivial_feature_r2(
            train,
            test,
            n_tokens=int(np.asarray(emission_matrix).shape[1]),
            fit=fit,
            rank=rank,
        ),
        **residual_r2_beyond_recent_tokens(predicted, test),
    }
