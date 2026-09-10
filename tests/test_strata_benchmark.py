from __future__ import annotations

from itertools import product
import json

import numpy as np
import pytest

from envs.hmm import HMMEnv
from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_1.benchmark import (
    bayes_accuracy_bounds,
    exactly_stationary_suffix_bounds,
    finite_episode_accuracy_bounds,
    monte_carlo_bayes_accuracy,
    pending_token_geometry,
    stratified_stationary_bounds,
)
from experiments.strata_token_guess_cycle_1.process import environment_config


def _exhaustive_latent_paths(depth, model):
    lower = 0.0
    upper = 0.0
    mass = 0.0
    switch_mass = 0.0
    for context in product(range(2), repeat=depth):
        source_joint = np.zeros((3, 2))
        for token in range(2):
            for states in product(range(3), repeat=depth + 2):
                probability = model.initial_distribution[states[0]]
                for step, observed in enumerate((*context, token)):
                    probability *= model.edge_transition_matrices[
                        observed, states[step], states[step + 1]
                    ]
                source_joint[states[0], token] += probability
        joint = source_joint.sum(axis=0)
        lower += joint.max()
        upper += source_joint.max(axis=1).sum()
        mass += joint.sum()
        if joint[0] > joint[1]:
            switch_mass += joint.sum()
    return lower, upper, mass, switch_mass


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("batch_depth", [0, 1, 3])
def test_suffix_enumeration_matches_exhaustive_latent_and_token_paths(depth, batch_depth):
    expected = _exhaustive_latent_paths(depth, strata_model(**environment_config()["model"]["kwargs"]))
    actual = exactly_stationary_suffix_bounds(depth, batch_depth=batch_depth)
    np.testing.assert_allclose(
        [actual["lower"], actual["upper"], actual["context_probability_mass"], actual["strict_switch_probability"]],
        expected,
        atol=2e-14,
        rtol=0.0,
    )
    json.dumps(actual, allow_nan=False)


def test_published_model_stationarity_constant_baseline_and_source_genie():
    model = strata_model(**environment_config()["model"]["kwargs"])
    np.testing.assert_allclose(model.initial_distribution @ model.transition_matrix, model.initial_distribution)
    np.testing.assert_allclose(model.emission_matrix[:, 0], [0.3686, 0.5238, 0.0])
    result = exactly_stationary_suffix_bounds(0)
    assert result["lower"] == pytest.approx(0.7025333333333333)
    assert result["upper"] == pytest.approx(0.7184)
    assert result["constant_guess"] == 1
    assert result["strict_switch_probability"] == 0.0


def test_bayes_action_switches_after_five_zeros_and_is_not_constant_token_rule():
    model = strata_model(**environment_config()["model"]["kwargs"])
    source = model.initial_distribution.copy()
    for _ in range(4):
        source = source @ model.edge_transition_matrices[0]
        source /= source.sum()
    assert (source @ model.emission_matrix).argmax() == 1
    source = source @ model.edge_transition_matrices[0]
    source /= source.sum()
    assert (source @ model.emission_matrix).argmax() == 0
    assert source[2] == 0.0
    result = exactly_stationary_suffix_bounds(5)
    assert result["benefit_over_constant"] == pytest.approx(2.9576316712853767e-5)
    assert result["strict_switch_probability"] == pytest.approx(0.015411418810460435)


def test_suffix_bounds_are_monotone_and_batch_partition_invariant():
    results = [exactly_stationary_suffix_bounds(depth, batch_depth=2) for depth in range(11)]
    assert np.all(np.diff([row["lower"] for row in results]) >= -1e-14)
    assert np.all(np.diff([row["upper"] for row in results]) <= 1e-14)
    whole = exactly_stationary_suffix_bounds(10, batch_depth=10)
    np.testing.assert_allclose(
        [results[-1]["lower"], results[-1]["upper"]],
        [whole["lower"], whole["upper"]],
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize("parameters", [
    {"alpha": 0.8, "t0": 0.2, "t1": 0.4},
    {"alpha": 0.97, "t0": 0.38, "t1": 0.38},
])
def test_renewal_bounds_recover_exact_constant_optimum(parameters):
    model = strata_model(**parameters)
    exact = float(np.max(model.initial_distribution @ model.emission_matrix))
    result = stratified_stationary_bounds(**parameters, grid_size=33)
    assert result["lower"] - 2e-12 <= exact <= result["upper"] + 2e-12
    assert result["gap"] < 1e-10


def test_default_bayes_reference_meets_tolerance_and_keeps_finite_scope_distinct():
    result = bayes_accuracy_bounds()
    assert result["tolerance_met"]
    assert 0.0 <= result["gap"] <= result["tolerance"] == 1e-5
    assert 0.70342 < result["lower"] < result["upper"] < 0.70344
    assert result["reference_accuracy"] == result["upper"]
    assert result["actual_depth"] == 20
    assert result["suffix_lower"] < result["lower"]
    assert result["finite_prefix_lower"] == result["suffix_lower"]
    assert result["finite_prefix_upper"] == result["upper"]
    assert result["benefit_over_constant_lower"] > 0.00088
    assert result["renewal_refinement"]["tail_probability_upper"] <= 1e-13
    assert "not interval" in result["numerics"]
    json.dumps(result, allow_nan=False)


def test_renewal_residual_bounds_do_not_assume_grid_iteration_convergence():
    accurate = stratified_stationary_bounds(grid_size=513)
    unfinished = stratified_stationary_bounds(grid_size=33, max_iterations=1, tail_tolerance=1e-3)
    assert unfinished["lower"] <= accurate["lower"]
    assert unfinished["upper"] >= accurate["upper"]
    assert unfinished["gap"] > accurate["gap"]


def test_stationary_reset_has_no_latent_mixing_transient():
    model = strata_model(**environment_config()["model"]["kwargs"])
    prior = model.initial_distribution
    np.testing.assert_array_equal(prior @ model.transition_matrix, prior)
    np.testing.assert_allclose(prior @ np.linalg.matrix_power(model.transition_matrix, 32), prior, rtol=0, atol=1e-14)
    exact_first_guess = float(np.max(prior @ model.emission_matrix))
    suffix = exactly_stationary_suffix_bounds(0)
    assert suffix["lower"] == pytest.approx(exact_first_guess)
    assert suffix["upper"] > exact_first_guess
    bound = bayes_accuracy_bounds(max_depth=5)
    first_decision = finite_episode_accuracy_bounds(bound, horizon=1, warmup=0)
    assert first_decision["lower"] == first_decision["upper"] == pytest.approx(exact_first_guess)


def test_finite_episode_prefixes_before_suffix_depth_are_exact():
    benchmark = bayes_accuracy_bounds(max_depth=5)
    exact = np.mean([exactly_stationary_suffix_bounds(depth)["lower"] for depth in range(4)])
    result = finite_episode_accuracy_bounds(benchmark, horizon=4, warmup=0)
    assert result["lower"] == pytest.approx(exact)
    assert result["upper"] == pytest.approx(exact)
    assert result["gap"] == 0.0
    matured = finite_episode_accuracy_bounds(benchmark, horizon=1024, warmup=32)
    assert matured["lower"] == pytest.approx(benchmark["finite_prefix_lower"])
    assert matured["upper"] == pytest.approx(benchmark["finite_prefix_upper"])
    assert matured["scored_steps_per_episode"] == 992


def test_unattained_tolerance_and_unsupported_renewal_are_reported_honestly():
    result = bayes_accuracy_bounds(max_depth=0, renewal_grid_size=3, tolerance=1e-14)
    assert not result["tolerance_met"]
    assert result["gap"] > result["tolerance"]
    unsupported = bayes_accuracy_bounds(alpha=1.0, max_depth=2)
    assert unsupported["renewal_refinement"] is None
    assert unsupported["renewal_error"]
    assert unsupported["reference_accuracy"] == unsupported["suffix_upper"]


def test_constant_optimum_can_stop_at_depth_zero():
    result = bayes_accuracy_bounds(alpha=0.8, t0=0.2, t1=0.4)
    assert result["actual_depth"] == 0
    assert result["gap"] == pytest.approx(0.0)
    assert result["renewal_refinement"] is None
    assert result["lower"] == pytest.approx(0.84)


def test_pending_geometry_uses_inverse_transition_not_arrival_emissions():
    model = strata_model(**environment_config()["model"]["kwargs"])
    geometry = pending_token_geometry()
    pending_map = np.asarray(geometry["arrival_to_pending_token_map"])
    direction = np.asarray(geometry["arrival_null_direction"])
    beliefs = np.random.default_rng(12).dirichlet(np.ones(3), size=100)
    arrival = beliefs @ model.transition_matrix
    np.testing.assert_allclose(arrival @ pending_map, beliefs @ model.emission_matrix, atol=2e-15)
    assert np.max(np.abs(arrival @ model.emission_matrix - beliefs @ model.emission_matrix)) > 0.005
    np.testing.assert_allclose(direction @ pending_map, 0.0, atol=1e-15)
    assert abs(direction.sum()) < 1e-15
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert abs(np.dot(direction, [1.0, 0.0, -1.0]) / np.sqrt(2.0)) < 0.9
    np.testing.assert_allclose(direction, geometry["source_null_direction"], atol=1e-15)
    assert pending_map.min() < 0.0
    json.dumps(geometry, allow_nan=False)


def test_public_environment_delay_one_belief_matches_source_times_transition():
    config = environment_config()
    config["randomize_first_episode_length"] = False
    config["diagnostics"] = {"belief": True, "tokens": True}
    env = HMMEnv(config)
    try:
        observation, info = env.reset(seed=17)
        model = env.model
        source = model.initial_distribution.copy()
        pending_map = np.asarray(pending_token_geometry()["arrival_to_pending_token_map"])
        assert not observation.any()
        for _ in range(128):
            arrival = np.asarray(info["belief_current"])
            np.testing.assert_allclose(arrival, source @ model.transition_matrix, atol=1e-14)
            np.testing.assert_allclose(arrival @ pending_map, source @ model.emission_matrix, atol=1e-14)
            token_before = info["raw_token_current"]
            guess = int(np.argmax(source @ model.emission_matrix))
            observation, reward, terminated, truncated, info = env.step(guess)
            assert reward == float(guess == token_before)
            assert not terminated and not truncated
            delivered = int(np.argmax(observation))
            assert delivered == token_before
            source = source @ model.edge_transition_matrices[delivered]
            source /= source.sum()
    finally:
        env.close()


def test_monte_carlo_is_reproducible_clustered_and_has_exact_stationary_control_variate():
    arguments = {"episodes": 64, "horizon": 128, "warmup": 16, "seed": 18}
    result = monte_carlo_bayes_accuracy(**arguments)
    assert result == monte_carlo_bayes_accuracy(**arguments)
    assert result["scored_steps"] == 64 * 112
    benefit = result["conditional_benefit_over_constant"]
    controlled = result["constant_control_variate_bayes_accuracy"]
    assert controlled["mean"] == pytest.approx(0.7025333333333333 + benefit["mean"])
    assert controlled["episode_cluster_standard_error"] == pytest.approx(benefit["episode_cluster_standard_error"])
    assert 0.0 < result["strict_switch_probability"]["mean"] < 0.2
    assert "not an exact" in result["warning"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("arguments", [
    {"depth": -1}, {"depth": 25}, {"depth": True}, {"depth": 1.5},
    {"depth": 2, "batch_depth": -1}, {"depth": 2, "alpha": float("nan")},
])
def test_invalid_suffix_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        exactly_stationary_suffix_bounds(**arguments)


@pytest.mark.parametrize("arguments", [
    {"tolerance": 0.0}, {"tolerance": float("nan")}, {"max_depth": 25},
])
def test_invalid_benchmark_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        bayes_accuracy_bounds(**arguments)


def test_invalid_monte_carlo_episode_and_warmup_inputs_are_rejected():
    with pytest.raises(ValueError):
        monte_carlo_bayes_accuracy(episodes=1)
    with pytest.raises(ValueError):
        monte_carlo_bayes_accuracy(horizon=32, warmup=32)
