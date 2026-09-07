from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from experiments.wing_two_factor_explore_cycle_1 import design


def _independent_kernels(alpha=0.94, x=0.4, strength=0.15):
    b = (1 - alpha) / 2
    edges = np.array([
        [[0, b, 0], [0, x * alpha, b / 2], [b, 0, 0]],
        [[alpha, 0, b], [b, (1 - x) * alpha, b / 2], [0, b, alpha]],
    ])
    result = np.zeros((3, 2, 3, 3))
    for action, token, source, arrival in itertools.product(range(3), range(2), range(3), range(3)):
        if action == 0:
            result[action, token, source, arrival] = edges[token, source, arrival]
        else:
            shift = 1 if action == 1 else -1
            result[action, token, source, arrival] = (
                (1 - strength) * edges[token, source, arrival]
                + strength * edges[token, source, (arrival - shift) % 3]
            )
    return result


def _stationary_by_iteration(transition):
    probability = np.full(len(transition), 1 / len(transition))
    for _ in range(20000):
        updated = probability @ transition
        if np.max(np.abs(updated - probability)) < 1e-14:
            return updated
        probability = updated
    raise AssertionError("independent stationary iteration did not converge")


@pytest.mark.parametrize("alpha,x,strength", [(0.94, 0.4, 0.15), (0.8, 0.25, 0.3)])
def test_shared_wing_kernels_match_independent_edge_and_control_formula(alpha, x, strength):
    expected = _independent_kernels(alpha, x, strength)
    actual = design.controlled_kernels(alpha=alpha, x=x, strength=strength)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(actual.sum(axis=(1, 3)), 1, atol=1e-15)
    np.testing.assert_allclose(actual.sum(axis=(1, 2)), 1, atol=1e-15)
    assert (actual >= 0).all()
    for transition in actual.sum(axis=1):
        np.testing.assert_allclose(design.stationary_distribution(transition), np.full(3, 1 / 3))


def test_all_nine_reactive_policies_use_current_token_augmented_chain():
    kernels = _independent_kernels()
    baseline = design.exact_baselines(kernels)
    assert len(baseline["reactive_policies"]) == 9
    for row in baseline["reactive_policies"]:
        policy = row["policy"]
        expected = np.zeros((6, 6))
        for state, token, arrival, next_token in itertools.product(range(3), range(2), range(3), range(2)):
            expected[2 * state + token, 2 * arrival + next_token] = kernels[
                policy[token], next_token, state, arrival
            ]
        np.testing.assert_array_equal(design.reactive_transition(kernels, tuple(policy)), expected)
        occupancy = _stationary_by_iteration(expected).reshape(3, 2).sum(axis=1)
        np.testing.assert_allclose(row["occupancy"], occupancy, atol=1e-12)
        joint = design.stationary_distribution(expected)
        expected_arrival_reward = joint @ expected @ np.repeat(np.eye(3), 2, axis=0)
        np.testing.assert_allclose(row["occupancy"], expected_arrival_reward, atol=1e-12)


def test_all_27_oracle_policies_and_average_reward_upper_bound():
    kernels = _independent_kernels()
    transitions = kernels.sum(axis=1)
    baseline = design.exact_baselines(kernels)
    assert len(baseline["oracle_policies"]) == 27
    for row in baseline["oracle_policies"]:
        transition = np.array([transitions[action, state] for state, action in enumerate(row["policy"])])
        np.testing.assert_allclose(row["occupancy"], _stationary_by_iteration(transition), atol=1e-12)
    for reward_state in range(3):
        best = baseline["reward_states"][str(reward_state)]
        policy = best["oracle_policy"]
        transition = np.array([transitions[action, state] for state, action in enumerate(policy)])
        reward = np.eye(3)[reward_state]
        gain = best["oracle_upper_bound"]
        bias = np.linalg.solve(np.eye(3) - transition + np.ones((3, 3)) / 3, transition @ reward - gain)
        advantages = np.array([
            [transitions[action, state] @ (reward + bias) - bias[state] for action in range(3)]
            for state in range(3)
        ])
        np.testing.assert_allclose(advantages.max(axis=1), gain, atol=1e-12)


def test_baseline_numeric_regression_and_analytic_json_has_no_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("analytic summary must not simulate")
    monkeypatch.setattr(design, "simulate_policy", forbidden)
    summary = design.design_summary()
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    expected = [(0.4406876289083306, [2, 0]), (0.39782569134742946, [0, 1]), (0.45424557527647635, [1, 0])]
    for state, (occupancy, policy) in enumerate(expected):
        baseline = summary["baselines"]["reward_states"][str(state)]
        assert baseline["best_constant"] == pytest.approx(1 / 3, abs=1e-12)
        assert baseline["best_reactive"] == pytest.approx(occupancy, abs=1e-12)
        assert baseline["best_reactive_policy"] == policy
        assert baseline["oracle_upper_bound"] == pytest.approx(0.7350993377483442, abs=1e-12)


@pytest.mark.parametrize("reward_state", range(3))
def test_discounted_q_uses_arrival_reward_and_satisfies_bellman_equation(reward_state):
    kernels = _independent_kernels()
    transitions = kernels.sum(axis=1)
    q = design.fully_observed_q_values(kernels, reward_state)
    for state, action in itertools.product(range(3), range(3)):
        expected = sum(transitions[action, state, j] * ((j == reward_state) + 0.99 * max(q[j])) for j in range(3))
        assert q[state, action] == pytest.approx(expected, abs=2e-12)
    np.testing.assert_allclose(
        design.fully_observed_q_values(kernels, reward_state, gamma=0),
        transitions[:, :, reward_state].T,
    )


@pytest.mark.parametrize("reward_state", range(3))
def test_default_qmdp_and_myopic_have_identical_belief_action_rankings(reward_state):
    kernels = _independent_kernels()
    q = design.fully_observed_q_values(kernels, reward_state)
    myopic = kernels.sum(axis=1)[:, :, reward_state].T
    contrast = (0.94 - 0.03) * (1 - 0.15)
    multiplier = 1 / (1 - 0.99 * contrast)
    difference = q - multiplier * myopic
    np.testing.assert_allclose(difference, np.full((3, 3), difference[0, 0]), atol=2e-12)


def test_null_is_preserved_by_all_one_step_emission_maps_but_not_all_future_tokens():
    kernels = _independent_kernels()
    delta = np.array([1, 0, -1]) / np.sqrt(2)
    for action in range(3):
        emission = kernels[action].sum(axis=-1).T
        np.testing.assert_allclose(delta @ emission, 0, atol=2e-16)
        assert np.linalg.matrix_rank(emission) == 2
    maximum = max(
        abs(delta @ kernels[action, token] @ kernels[next_action, next_token].sum(axis=-1))
        for action, token, next_action, next_token in itertools.product(range(3), range(2), range(3), range(2))
    )
    assert maximum == pytest.approx(0.041353018777351744)
    report = design.identifiability_summary(kernels)
    assert report["max_two_step_token_difference"] == pytest.approx(maximum)
    assert report["max_one_step_null_residual"] < 2e-16


def test_bayes_filter_matches_independent_hidden_path_enumeration():
    kernels = _independent_kernels()
    actions, tokens = np.array([1, 0, 2, 1]), np.array([1, 0, 1, 0])
    masses = np.zeros(3)
    for path in itertools.product(range(3), repeat=5):
        probability = 1 / 3
        for step in range(4):
            probability *= kernels[actions[step], tokens[step], path[step], path[step + 1]]
        masses[path[-1]] += probability
    np.testing.assert_allclose(design.suffix_belief(kernels, actions, tokens), masses / masses.sum(), atol=1e-14)
    reversed_controls = design.suffix_belief(kernels, np.array([2, 0, 1, 2]), tokens)
    assert not np.allclose(reversed_controls, masses / masses.sum())


@pytest.mark.parametrize("length", [1, 2, 7, 32])
def test_rolling_window_restarts_before_oldest_action_token_and_matches_exact_replay(length):
    rng = np.random.default_rng(12)
    kernels = _independent_kernels()
    actions, tokens = rng.integers(3, size=(101, 5)), rng.integers(2, size=(101, 5))
    window = design._WindowFilter(length, 5)
    for step in range(len(actions)):
        actual = window.update(kernels[actions[step], tokens[step]])
        start = max(0, step + 1 - length)
        expected = design.suffix_belief(kernels, actions[start:step + 1], tokens[start:step + 1])
        np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_independent_reactive_monte_carlo_agrees_with_augmented_stationary_baseline():
    kernels = _independent_kernels()
    rng = np.random.default_rng(34)
    chains = 512
    states, tokens = rng.integers(3, size=chains), np.zeros(chains, dtype=int)
    policy = np.array([2, 0])
    occupancy = np.zeros(chains)
    for step in range(2000):
        cumulative = kernels[policy[tokens], :, states].reshape(chains, 6).cumsum(axis=1)
        outcomes = (rng.random((chains, 1)) > cumulative).sum(axis=1)
        tokens, states = outcomes // 3, outcomes % 3
        if step >= 300:
            occupancy += states == 0
    means = occupancy / 1700
    stderr = means.std(ddof=1) / np.sqrt(chains)
    assert abs(means.mean() - 0.4406876289083306) < 5 * stderr


def test_optional_audit_is_deterministic_json_and_reports_chainwise_uncertainty():
    first = design.demand_audit(seed=21, chains=8, steps=40, burn_in=8)
    second = design.demand_audit(seed=21, chains=8, steps=40, burn_in=8)
    assert first == second
    json.dumps(first, allow_nan=False)
    for reward_state, policies in first["reward_states"].items():
        baseline = first["design"]["baselines"]["reward_states"][reward_state]
        for result in policies.values():
            occupancy = result["occupancy"]
            assert 0 <= occupancy["mean"] <= 1
            assert result["gap_standard_error"] == occupancy["standard_error"]
            assert result["gap_over_best_reactive"] == pytest.approx(occupancy["mean"] - baseline["best_reactive"])
            for mse in result["suffix_mse"].values():
                assert np.shape(mse["mean"]) == (4,)
                assert (np.array(mse["mean"]) >= 0).all()
    estimate = design._estimate(np.array([0.1, 0.3, 0.5]))
    assert estimate["mean"] == pytest.approx(0.3)
    assert estimate["standard_error"] == pytest.approx(0.2 / np.sqrt(3))


@pytest.mark.parametrize("kwargs", [{"chains": 1}, {"burn_in": 10}, {"burn_in": -1}])
def test_simulation_rejects_invalid_sample_counts(kwargs):
    arguments = {"seed": 0, "chains": 2, "steps": 10, "burn_in": 0} | kwargs
    with pytest.raises(ValueError):
        design.simulate_policy(_independent_kernels(), 0, "full_qmdp", **arguments)
