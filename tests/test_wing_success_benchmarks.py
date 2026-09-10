from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
from scipy.stats import t as student_t

from experiments.wing_two_factor_explore_cycle_1 import success_benchmarks as benchmarks


def _edges(alpha=0.94, x=0.4):
    beta = (1 - alpha) / 2
    return np.array([
        [[0, beta, 0], [0, x * alpha, beta / 2], [beta, 0, 0]],
        [[alpha, 0, beta], [beta, (1 - x) * alpha, beta / 2], [0, beta, alpha]],
    ])


def _kernels(alpha=0.94, x=0.4, strength=1.0):
    base = _edges(alpha, x)
    return np.stack([
        base,
        (1 - strength) * base + strength * np.roll(base, 1, axis=-1),
        (1 - strength) * base + strength * np.roll(base, -1, axis=-1),
    ])


def _exact_value(belief, horizon, kernels, reward_state):
    if horizon == 0:
        return 0.0
    candidates = []
    for action in range(3):
        value = belief @ kernels[action].sum(axis=0)[:, reward_state]
        for token in range(2):
            mass = belief @ kernels[action, token]
            probability = mass.sum()
            if probability > 0:
                value += probability * _exact_value(mass / probability, horizon - 1, kernels, reward_state)
        candidates.append(value)
    return max(candidates)


def _exact_reset_value(horizon, kernels, reward_state):
    value = 0.0
    for token in range(2):
        mass = np.full(3, 1 / 3) @ kernels[0, token]
        probability = mass.sum()
        if probability > 0:
            value += probability * _exact_value(mass / probability, horizon, kernels, reward_state)
    return value


def _exact_passive_accuracy(alpha, x, horizon):
    edges = _edges(alpha, x)
    emission = edges.sum(axis=-1).T
    masses = [np.full(3, 1 / 3)]
    total = 0.0
    for _ in range(horizon):
        one_factor = sum((mass @ emission).max() for mass in masses)
        total += one_factor ** 2
        masses = [mass @ edges[token] for mass in masses for token in range(2)]
    return total / horizon


@pytest.mark.parametrize("subdivisions", [1, 2, 7, 100])
def test_simplex_stencils_reconstruct_corners_boundaries_grid_and_interior(subdivisions):
    grid = benchmarks.simplex_grid(subdivisions)
    assert len(grid) == (subdivisions + 1) * (subdivisions + 2) // 2
    np.testing.assert_allclose(grid.sum(axis=1), 1, atol=2e-16)
    assert np.all(grid >= 0)
    line = np.linspace(0, 1, 501)
    boundary = np.column_stack((line, 1 - line, np.zeros(len(line))))
    beliefs = np.concatenate([
        grid, np.eye(3), boundary, boundary[:, [2, 0, 1]], boundary[:, [1, 2, 0]],
        np.random.default_rng(12).dirichlet(np.ones(3), size=5000),
    ])
    indices, weights = benchmarks.simplex_stencil(beliefs, subdivisions)
    assert indices.shape == weights.shape == beliefs.shape
    assert np.all(indices >= 0) and np.all(indices < len(grid))
    assert np.all(weights >= 0) and np.all(weights <= 1)
    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=3e-16)
    np.testing.assert_allclose((grid[indices] * weights[..., None]).sum(axis=-2), beliefs, atol=1e-14)
    affine = grid @ np.array([0.8, -0.4, 0.1])
    np.testing.assert_allclose(benchmarks._interpolate(affine, indices, weights), beliefs @ np.array([0.8, -0.4, 0.1]), atol=1e-14)
    convex = (grid ** 2).sum(axis=-1)
    assert np.all(benchmarks._interpolate(convex, indices, weights) >= (beliefs ** 2).sum(axis=-1) - 1e-14)


@pytest.mark.parametrize("belief", [[-0.01, 0.5, 0.51], [0.2, 0.3, 0.4], [np.nan, 0, 1], [0.2, 0.8]])
def test_simplex_stencil_rejects_extrapolation(belief):
    with pytest.raises(ValueError, match="simplex"):
        benchmarks.simplex_stencil(np.array(belief), 20)


def test_grid_sparse_bellman_has_exact_arrival_reward_and_valid_probabilities():
    planner = benchmarks._WingGrid(0.94, 0.4, 1.0, 0, 20)
    kernels = _kernels()
    np.testing.assert_allclose(planner.kernels, kernels, atol=1e-16)
    expected_reward = np.array([[belief @ kernels[action].sum(axis=0)[:, 0] for action in range(3)] for belief in planner.grid])
    np.testing.assert_allclose(planner.rewards, expected_reward, atol=1e-16)
    np.testing.assert_allclose(planner.q_values(np.zeros(len(planner.grid))), expected_reward)
    np.testing.assert_allclose(np.asarray(planner.transition.sum(axis=1)).ravel(), 1, atol=1e-15)
    assert np.all(planner.transition.data >= 0)
    assert planner.reconstruction_error < 1e-14
    affine = planner.grid @ np.array([0.2, 0.7, -0.4])
    expected_future = np.einsum("bi,aij,j->ba", planner.grid, kernels.sum(axis=1), np.array([0.2, 0.7, -0.4]))
    np.testing.assert_allclose(planner.q_values(affine) - planner.rewards, expected_future, atol=1e-15)


@pytest.mark.parametrize("reward_state", [0, 1, 2])
@pytest.mark.parametrize("strength", [0.0, 0.15, 1.0])
def test_one_step_bound_is_exact_reset_expected_reward_without_initial_grid_revelation(reward_state, strength):
    exact = _exact_reset_value(1, _kernels(strength=strength), reward_state)
    for subdivisions in [1, 7, 20]:
        result = benchmarks.controlled_bayes_bound(horizon=1, strength=strength, reward_state=reward_state, subdivisions=subdivisions)
        assert result["upper_bound_fraction"] == pytest.approx(exact, abs=1e-15)
        assert result["episode_return_upper_bound"] == result["upper_bound_fraction"]
        np.testing.assert_allclose(np.array(result["reset_token_probabilities"]) @ np.array(result["reset_beliefs"]), np.full(3, 1 / 3))


@pytest.mark.parametrize("horizon", [2, 3, 4])
def test_bound_dominates_exhaustive_policy_tree_and_refines_for_study_parameters(horizon):
    exact = _exact_reset_value(horizon, _kernels(), 0)
    levels = [2, 4, 8, 16, 32]
    bounds = [benchmarks.controlled_bayes_bound(horizon=horizon, subdivisions=level)["episode_return_upper_bound"] for level in levels]
    assert all(bound >= exact - 2e-14 for bound in bounds)
    assert all(second <= first + 2e-14 for first, second in zip(bounds, bounds[1:]))
    assert bounds[-1] - exact < 0.02


def test_corner_grid_equals_full_state_revelation_after_first_controlled_edge():
    kernels = _kernels()
    values = np.zeros(3)
    for _ in range(4):
        values = np.einsum("aij,j->ai", kernels.sum(axis=1), np.eye(3)[0] + values).max(axis=0)
    total = 0.0
    for token in range(2):
        mass = np.full(3, 1 / 3) @ kernels[0, token]
        total += max(mass @ transition @ (np.eye(3)[0] + values) for transition in kernels.sum(axis=1))
    actual = benchmarks.controlled_bayes_bound(horizon=5, subdivisions=1)
    assert actual["episode_return_upper_bound"] == pytest.approx(total, abs=2e-14)


def test_total_value_increases_with_horizon_and_has_at_most_one_additional_reward():
    values = [benchmarks.controlled_bayes_bound(horizon=horizon, subdivisions=12)["episode_return_upper_bound"] for horizon in range(1, 8)]
    assert np.all(np.diff(values) >= -1e-14)
    assert np.all(np.diff(values) <= 1 + 1e-14)


def test_refinement_report_selects_computed_minimum_without_extrapolation():
    report = benchmarks.controlled_bayes_refinements(horizon=10, subdivisions=(7, 14, 28))
    bounds = [row["upper_bound_fraction"] for row in report["refinements"]]
    assert report["upper_bound_fraction"] == min(bounds)
    assert report["selected_subdivisions"] == 28
    assert report["convergence"]["nested_triangulations"] is True
    assert report["convergence"]["observed_nonincreasing"] is True
    np.testing.assert_allclose(report["convergence"]["successive_fraction_changes"], np.diff(bounds))
    assert report["convergence"]["last_absolute_fraction_change"] == abs(bounds[-1] - bounds[-2])
    assert "not a certified error" in report["convergence"]["interpretation"]
    other = benchmarks.controlled_bayes_refinements(horizon=2, subdivisions=(7, 10))
    assert other["convergence"]["nested_triangulations"] is False
    json.dumps(report, allow_nan=False)


def test_independent_factors_mean_reward_has_single_factor_optimum_not_squared():
    kernels = _kernels()
    total = 0.0
    for token_1, token_2 in itertools.product(range(2), repeat=2):
        mass_1 = np.full(3, 1 / 3) @ kernels[0, token_1]
        mass_2 = np.full(3, 1 / 3) @ kernels[0, token_2]
        probability = mass_1.sum() * mass_2.sum()
        belief_1, belief_2 = mass_1 / mass_1.sum(), mass_2 / mass_2.sum()
        scores = []
        for action_1, action_2 in itertools.product(range(3), repeat=2):
            first = belief_1 @ kernels[action_1].sum(axis=0)[:, 0]
            second = belief_2 @ kernels[action_2].sum(axis=0)[:, 0]
            scores.append((first + second) / 2)
        total += probability * max(scores)
    exact_single = _exact_reset_value(1, kernels, 0)
    assert total == pytest.approx(exact_single, abs=1e-15)
    assert abs(total - exact_single ** 2) > 0.1


@pytest.mark.parametrize("alpha,x", [(0.94, 0.4), (0.94, 0.9), (1.0, 1.0)])
def test_passive_blank_first_decision_is_exact_unconditioned_joint_prediction(alpha, x):
    expected = (np.full(3, 1 / 3) @ _edges(alpha, x).sum(axis=-1).T).max() ** 2
    result = benchmarks.token_guess_bayes_accuracy(alpha=alpha, x=x, horizon=1, n_episodes=16, seed=8)
    assert result["mean"] == pytest.approx(expected, abs=1e-15)
    assert result["first_decision_accuracy"] == pytest.approx(expected, abs=1e-15)
    assert result["standard_error"] < 1e-15
    assert result["warmup_actions"] == 0
    assert result["reset_count"] == 16


@pytest.mark.parametrize("alpha,x", [(0.94, 0.4), (0.94, 0.9)])
def test_passive_monte_carlo_agrees_with_exhaustive_observation_sequences(alpha, x):
    exact = _exact_passive_accuracy(alpha, x, 4)
    result = benchmarks.token_guess_bayes_accuracy(alpha=alpha, x=x, horizon=4, n_episodes=32768, seed=98)
    assert abs(result["mean"] - exact) < 6 * result["standard_error"]
    assert result["expected_episode_return"] == result["mean"] * 4
    if x == 0.9:
        assert result["analytic_cross_check"] is None
        assert exact > (np.full(3, 1 / 3) @ _edges(alpha, x).sum(axis=-1).T).max() ** 2


def test_passive_default_has_exact_statewise_dominant_token_analytic_solution():
    result = benchmarks.token_guess_bayes_accuracy(horizon=19, n_episodes=32, seed=8)
    analytic = result["analytic_cross_check"]
    np.testing.assert_allclose(analytic["emission_probabilities_of_dominant_token"], [0.97, 0.609, 0.97])
    assert analytic["joint_guess"] == [1, 1]
    assert analytic["fraction"] == pytest.approx(((0.97 + 0.609 + 0.97) / 3) ** 2, abs=1e-14)
    assert analytic["fraction"] == pytest.approx(_exact_passive_accuracy(0.94, 0.4, 5), abs=1e-14)


def test_passive_seeded_reference_uses_source_belief_not_one_extra_transition():
    alpha, x, horizon, episodes, seed = 0.94, 0.9, 7, 13, 12
    edges = _edges(alpha, x)
    emission = edges.sum(axis=-1).T
    random = np.random.default_rng(seed).random((horizon, episodes, 2))
    means = []
    for episode in range(episodes):
        factors = [np.full(3, 1 / 3), np.full(3, 1 / 3)]
        score = 0.0
        for step in range(horizon):
            probabilities = np.array([belief @ emission for belief in factors])
            score += probabilities[0].max() * probabilities[1].max()
            for factor in range(2):
                token = int(random[step, episode, factor] >= probabilities[factor, 0])
                mass = factors[factor] @ edges[token]
                factors[factor] = mass / mass.sum()
        means.append(score / horizon)
    actual = benchmarks.token_guess_bayes_accuracy(alpha=alpha, x=x, horizon=horizon, n_episodes=episodes, seed=seed)
    expected = benchmarks._episode_estimate(np.array(means))
    assert actual["mean"] == pytest.approx(expected["mean"], abs=1e-15)
    assert actual["standard_error"] == pytest.approx(expected["standard_error"], abs=1e-15)
    np.testing.assert_allclose(actual["ci95"], expected["ci95"], atol=1e-15)


def test_episode_interval_clusters_whole_episodes_and_uses_sample_variance():
    means = np.array([0.1, 0.2, 0.5, 0.6])
    result = benchmarks._episode_estimate(means)
    se = means.std(ddof=1) / 2
    assert result["standard_error"] == pytest.approx(se)
    np.testing.assert_allclose(result["ci95"], means.mean() + np.array([-1, 1]) * student_t.ppf(0.975, 3) * se)


@pytest.mark.parametrize("alpha,x", [(1.0, 0.0), (1.0, 1.0), (1 - 1e-12, 0.4)])
def test_zero_probability_tokens_and_near_zero_beta_do_not_produce_nan(alpha, x):
    result = benchmarks.controlled_bayes_bound(alpha=alpha, x=x, horizon=8, subdivisions=12)
    assert 0 <= result["upper_bound_fraction"] <= 1 + 1e-14
    assert result["grid"]["minimum_barycentric_weight"] >= 0
    assert result["grid"]["max_posterior_reconstruction_error"] < 1e-14
    passive = benchmarks.token_guess_bayes_accuracy(alpha=alpha, x=x, horizon=8, n_episodes=16)
    json.dumps({"controlled": result, "passive": passive}, allow_nan=False)


def test_attainable_policy_monte_carlo_is_consistent_with_upper_bound_and_reproducible():
    settings = {"horizon": 64, "subdivisions": 32, "n_episodes": 512, "seed": 81}
    first = benchmarks.controlled_belief_policy_reward(**settings)
    second = benchmarks.controlled_belief_policy_reward(**settings)
    assert first == second
    bound = benchmarks.controlled_bayes_bound(horizon=64, subdivisions=64)
    assert first["mean"] <= bound["upper_bound_fraction"] + 6 * first["standard_error"]
    assert first["planning"]["bellman_increment_span"] <= first["planning"]["span_tolerance"]
    assert first["warmup_actions"] == 0
    assert first["reset_count"] == first["independent_chains"] == 512
    assert "no hidden-state or reward" in first["filter"]
    json.dumps(first, allow_nan=False)


def test_zero_discount_policy_matches_exact_seeded_myopic_episode_simulation():
    episodes, horizon, seed = 19, 11, 23
    kernels = _kernels()
    edges = kernels[0]
    reset_mass = np.einsum("i,xij->xj", np.full(3, 1 / 3), edges)
    reset_probability = reset_mass.sum(axis=-1)
    rng = np.random.default_rng(seed)
    reset_tokens = (rng.random(episodes) >= reset_probability[0]).astype(int)
    beliefs = (reset_mass / reset_probability[:, None])[reset_tokens]
    scores = np.zeros(episodes)
    for _ in range(horizon):
        rewards = beliefs @ kernels.sum(axis=1)[:, :, 0].T
        actions = rewards.argmax(axis=-1)
        scores += rewards[np.arange(episodes), actions]
        draws = rng.random(episodes)
        for index in range(episodes):
            mass = np.array([beliefs[index] @ kernels[actions[index], token] for token in range(2)])
            token = int(draws[index] >= mass[0].sum())
            beliefs[index] = mass[token] / mass[token].sum()
    actual = benchmarks.controlled_belief_policy_reward(horizon=horizon, n_episodes=episodes, seed=seed, subdivisions=9, gamma=0)
    assert actual["mean"] == pytest.approx((scores / horizon).mean(), abs=1e-14)
    assert actual["standard_error"] == pytest.approx((scores / horizon).std(ddof=1) / np.sqrt(episodes), abs=1e-14)


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"horizon": True}, {"subdivisions": 0}, {"reward_state": 3}, {"reward_state": False}, {"strength": 1.1}])
def test_bound_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        benchmarks.controlled_bayes_bound(**({"horizon": 2, "subdivisions": 4} | kwargs))


@pytest.mark.parametrize("kwargs", [{"n_episodes": 1}, {"seed": -1}, {"horizon": 1.5}, {"alpha": np.nan}])
def test_passive_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        benchmarks.token_guess_bayes_accuracy(**({"horizon": 2, "n_episodes": 4} | kwargs))


@pytest.mark.parametrize("levels", [(), (4, 4), (8, 4), (0, 4)])
def test_refinement_rejects_empty_or_unordered_levels(levels):
    with pytest.raises(ValueError):
        benchmarks.controlled_bayes_refinements(horizon=2, subdivisions=levels)


def test_stationary_policy_reports_nonconvergence_and_rejects_undiscounted_planning():
    with pytest.raises(RuntimeError, match="did not converge"):
        benchmarks.controlled_belief_policy_reward(horizon=2, subdivisions=4, n_episodes=4, max_iterations=1)
    with pytest.raises(ValueError, match="gamma"):
        benchmarks.controlled_belief_policy_reward(gamma=1)


def test_cli_prints_reproducible_finite_json(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["success_benchmarks", "--horizon", "3", "--episodes", "4", "--subdivisions", "4", "8", "--with-policy", "--policy-subdivisions", "8"])
    benchmarks.main()
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"passive", "controlled", "controlled_policy"}
    assert result["passive"]["horizon"] == 3
    assert result["controlled"]["selected_subdivisions"] == 8
    assert result["controlled_policy"]["planning"]["gamma"] == 0.99
