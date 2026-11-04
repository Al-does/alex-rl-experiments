from __future__ import annotations

import numpy as np
import pytest

from envs.mess3.model import control_model
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_logit_evidence import (
    analyze,
    condition,
    nuisance_features,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.design import (
    CYCLE_6_TRANSITION_MATRIX,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.task import (
    ActionSymmetryTask,
)


def test_reward_state_evidence_distinguishes_repetition_from_alternation():
    model = control_model(alpha=0.85, transition_matrix=CYCLE_6_TRANSITION_MATRIX)
    task = ActionSymmetryTask(model=model, variant=2)
    prior = np.tile(model.initial_distribution, (4, 1))
    first = condition(prior, model.emission_matrix, np.array([0, 0, 1, 1]))
    second = condition(
        first @ task.transition_matrix_for_action(1),
        model.emission_matrix, np.array([0, 1, 1, 0]),
    )
    np.testing.assert_allclose(second[:, 2], [
        0.0502721541118875, 0.11552828058291288,
        0.0502721541118875, 0.11552828058291288,
    ])


def synthetic(seed, slope):
    rng = np.random.default_rng(seed)
    n = 4096
    token = rng.integers(3, size=n)
    randomized = np.where(token == 2, 2, rng.integers(2, size=n))
    coarse = rng.uniform(0.05, 0.95, n)
    evidence = 0.01 * rng.normal(size=n) + 0.03 * (token - randomized)
    previous = rng.integers(2, size=n)
    nuisance = 2 * (token - randomized) + previous + coarse ** 2
    difference = nuisance + slope * evidence + 0.0001 * rng.normal(size=n)
    return {
        "episode": np.repeat(np.arange(8), n // 8),
        "step": np.tile(np.arange(n // 8) + 128, 8),
        "token": token,
        "randomized_token": randomized,
        "previous_action": previous,
        "coarse": coarse,
        "rho": coarse + evidence / 2,
        "randomized_rho": coarse - evidence / 2,
        "logit": difference,
        "randomized_logit": np.zeros(n),
        "total_variation": np.full(n, 0.001),
        "greedy_disagreement": np.zeros(n, dtype=bool),
    }


@pytest.mark.parametrize("slope", [1.3, -1.3])
def test_signed_heldout_test_does_not_assume_support(slope):
    report = analyze(synthetic(10, slope), synthetic(11, slope), resamples=20)
    for subset in ("all", "same_current_token", "current_C"):
        result = report[subset]
        assert result["fit_slope"] == pytest.approx(slope, abs=0.01)
        association = result["heldout_partial_association"]
        assert association["slope"] == pytest.approx(slope, abs=0.01)
        assert np.sign(association["correlation"]) == np.sign(slope)
        assert result["heldout_incremental_prediction"]["mse_improvement"] > 0


def test_nuisance_features_preserve_coarse_belief_and_current_token_pairs():
    data = synthetic(5, 1)
    design = nuisance_features(data)
    nuisance = (
        2 * (data["token"] - data["randomized_token"])
        + data["previous_action"] + data["coarse"] ** 2
    )
    weight = np.linalg.lstsq(design, nuisance, rcond=1e-10)[0]
    np.testing.assert_allclose(design @ weight, nuisance, atol=1e-10)
    data["previous_action"][0] = -1
    with pytest.raises(ValueError, match="post-reset"):
        nuisance_features(data)
