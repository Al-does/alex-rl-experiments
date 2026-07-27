"""Tests for the process-design scoring and the chosen operating point."""

from __future__ import annotations

import numpy as np
import pytest

from envs.hmm import HMMEnv
from envs.mess3.model import emission_matrix, symmetric_transition_matrix
from experiments.mess3_token_guess_cycle_1.operating_point_validation.experiment import (
    ARMS,
    build_config,
)
from experiments.mess3_token_guess_cycle_1.operating_points import (
    POINTS,
    PROPOSED,
    SHIPPED,
    point_by_name,
)
from experiments.mess3_token_guess_cycle_1.process_design import (
    best_windowed_accuracy,
    evaluate,
    simulate,
    spectral_gap,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _design(point, **kwargs):
    return evaluate(
        symmetric_transition_matrix(point.stay),
        emission_matrix(point.alpha),
        name=point.name,
        **kwargs,
    )


def test_spectral_gap_matches_the_closed_form_for_the_symmetric_chain():
    for stay in (0.5, 0.85, 0.90, 0.96):
        assert spectral_gap(symmetric_transition_matrix(stay)) == pytest.approx(
            (3.0 * stay - 1.0) / 2.0, abs=1e-12
        )


def test_a_deeper_token_window_never_scores_worse_than_a_shallower_one():
    transition = symmetric_transition_matrix(0.96)
    emission = emission_matrix(0.55)
    trajectory = simulate(transition, emission, n_chains=128, n_steps=240)
    scores = [
        best_windowed_accuracy(trajectory, depth, 3) for depth in (1, 2, 3)
    ]
    assert scores == sorted(scores)
    assert scores[-1] <= trajectory.predictive.max(axis=2).mean() + 1e-9


def test_a_full_rank_channel_lets_the_predictive_distribution_span_the_belief():
    design = _design(PROPOSED)
    assert design["probe_r2_sufficient"] == pytest.approx(1.0, abs=1e-6)


def test_the_proposed_point_widens_both_axes_against_the_shipped_one():
    shipped = _design(SHIPPED)
    proposed = _design(PROPOSED)

    assert proposed["accuracy_headroom"] > 3.0 * shipped["accuracy_headroom"]
    assert proposed["probe_band"] > 3.0 * shipped["probe_band"]
    assert proposed["probe_r2_window1"] < 0.5 < shipped["probe_r2_window1"]
    assert proposed["belief_conditioning"] > 0.9


def test_the_chain_outlasts_the_channel_at_every_named_point():
    """Below stay > alpha the Bayes guess is always the last token."""
    for point in POINTS:
        assert point.stay > point.alpha
        design = _design(point)
        assert design["accuracy_headroom"] > 0.0


def test_named_points_build_environments_the_harness_accepts():
    for point in POINTS:
        assert point_by_name(point.name) is point
        environment = HMMEnv(point.env_config(belief=True, tokens=True))
        try:
            np.testing.assert_allclose(
                np.diag(environment.model.transition_matrix),
                point.stay,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.diag(environment.model.emission_matrix),
                point.alpha,
                atol=1e-12,
            )
            assert environment.config.delay == 1
        finally:
            environment.close()
    with pytest.raises(ValueError, match="unknown operating point"):
        point_by_name("nope")


def test_validation_cells_differ_only_in_the_environment(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    configs = {
        (point.name, arm.name): build_config(context, point, arm.name)
        for point in POINTS
        for arm in ARMS
    }
    for config in configs.values():
        assert config.gamma == 0.99
        assert config.entropy_coeff == 0.0
        assert config.train_batch_size_per_learner == 2_048

    for arm in ARMS:
        shipped = configs[("shipped", arm.name)]
        proposed = configs[("proposed", arm.name)]
        assert shipped.env_config != proposed.env_config
        assert shipped.model_config == proposed.model_config
        assert shipped.vf_loss_coeff == proposed.vf_loss_coeff
