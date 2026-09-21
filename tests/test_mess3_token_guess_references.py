"""Tests for the token-guess metric reference points."""

from __future__ import annotations

import numpy as np
import pytest

from envs.mess3.model import passive_model
from experiments.mess3_token_guess_cycle_1.comparison.experiment import (
    BASE_MODEL_CONFIG,
    ENV_CONFIG,
)
from experiments.mess3_token_guess_cycle_2.metric_references import (
    ALPHA,
    bayes_accuracy_by_context,
    compute_references,
    normalise,
    raw_token_window_r2,
    simulate_stream,
    untrained_module_r2,
)


def test_simulated_belief_matches_the_delay_one_transducer_update():
    model = passive_model(alpha=ALPHA)
    transition = np.asarray(model.transition_matrix)
    emission = np.asarray(model.emission_matrix)
    stream = simulate_stream(n_steps=64, seed=3, window=8)

    for index in range(len(stream.beliefs) - 1):
        belief = stream.beliefs[index]
        token = stream.tokens[index]
        # delay=1 composes the observation operator before the transition, so
        # the update is diag(P(y|s)) @ T applied in row-vector convention.
        operator = np.diag(emission[:, token]) @ transition
        expected = belief @ operator
        expected /= expected.sum()
        np.testing.assert_allclose(stream.beliefs[index + 1], expected, atol=1e-12)


def test_simulated_beliefs_are_probability_vectors():
    stream = simulate_stream(n_steps=256, seed=11, window=8)
    np.testing.assert_allclose(stream.beliefs.sum(axis=1), 1.0, atol=1e-12)
    assert (stream.beliefs >= 0.0).all()


def test_raw_token_probe_saturates_well_below_the_supervised_ceiling():
    fit = simulate_stream(n_steps=20_000, seed=100)
    test = simulate_stream(n_steps=10_000, seed=101)
    scores = raw_token_window_r2(fit, test, context_lengths=(1, 2, 4, 8, 16))

    # More observations never hurt an affine probe by a meaningful margin.
    ordered = [scores[k] for k in (1, 2, 4, 8, 16)]
    assert all(later >= earlier - 5e-3 for earlier, later in zip(ordered, ordered[1:]))

    # A single observation is already most of the way there, and the window
    # saturates by eight observations, short of the supervised 0.9989.
    assert 0.78 < scores[1] < 0.83
    assert scores[8] == pytest.approx(scores[16], abs=5e-3)
    assert 0.95 < scores[8] < 0.98


def test_raw_token_floor_exceeds_several_published_cycle_1_scores():
    fit = simulate_stream(n_steps=20_000, seed=100)
    test = simulate_stream(n_steps=10_000, seed=101)
    floor = raw_token_window_r2(fit, test, context_lengths=(8,))[8]

    # Cycle 1 reported these for reward-only, max-entropy, and predictive-loss
    # PPO. A probe on the untransformed observations beats all three.
    for reported in (0.8552, 0.8558, 0.9319):
        assert reported < floor


def test_bayes_accuracy_spans_a_narrow_band_above_repeat_previous_token():
    stream = simulate_stream(n_steps=40_000, seed=102)
    accuracy = bayes_accuracy_by_context(stream, context_lengths=(1, 2, 4, 8, 64))

    # One observation is the repeat-the-previous-token rule.
    assert 0.66 < accuracy[1] < 0.69
    assert accuracy[8] == pytest.approx(accuracy[64], abs=5e-3)
    assert accuracy[64] - accuracy[1] < 0.03


def test_normalise_reports_position_within_the_usable_range():
    assert normalise(0.9670, floor=0.9670, ceiling=0.9989) == pytest.approx(0.0)
    assert normalise(0.9989, floor=0.9670, ceiling=0.9989) == pytest.approx(1.0)
    assert normalise(0.8552, floor=0.9670, ceiling=0.9989) < 0.0
    with pytest.raises(ValueError):
        normalise(0.5, floor=0.9, ceiling=0.9)


def test_untrained_module_is_probed_with_the_study_architecture():
    result = untrained_module_r2(
        env_config=ENV_CONFIG,
        model_config=BASE_MODEL_CONFIG,
        seed=5,
        fit_steps=2_000,
        test_steps=1_000,
    )
    assert result["n_fit"] == 2_000
    assert result["n_test"] == 1_000
    assert -1.0 <= result["r_squared"] <= 1.0
    assert 0.0 <= result["token_accuracy_greedy"] <= 1.0


def test_compute_references_reports_a_floor_below_the_supervised_ceiling():
    references = compute_references(seed=7, fit_steps=8_000, test_steps=4_000)
    floor = references["belief_r2_floor"]
    low, high = references["belief_r2_probe_noise_95ci"]

    assert 0.94 < floor < 0.99
    assert low < floor < high
    assert references["accuracy_floor_repeat_previous_token"] < (
        references["accuracy_ceiling_bayes"]
    )
    assert "untrained_module" not in references
