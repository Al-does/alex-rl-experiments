"""Tests for the targeted Cassandra exact-belief reward analysis."""

from __future__ import annotations

import numpy as np

from envs.cassandra_machine.model import N_STATES, TargetedAction
from experiments.cassandra_belief_factoring_2026_08.analyze_targeted_bayesian_reward import (
    bayes_update,
    build_model,
    finite_horizon_mdp_q,
    observable_after_next_values,
    simulate_bayesian_policy,
)


def test_reward_aware_bayes_update_conditions_before_transition():
    model = build_model()
    belief = np.full((1, N_STATES), 1.0 / N_STATES)
    action = np.array([int(TargetedAction.OPERATE)])
    observation = np.array([15])
    reward_class = np.array(
        [model.reward_class_by_state[N_STATES - 1]]
    )

    actual = bayes_update(
        model,
        belief,
        action,
        observation,
        observed_reward_classes=reward_class,
    )

    masked = belief[0] * model.reward_class_masks[reward_class[0]]
    expected = masked @ model.transitions[TargetedAction.OPERATE]
    expected *= model.observations[TargetedAction.OPERATE, :, 15]
    expected /= expected.sum()
    np.testing.assert_allclose(actual[0], expected, atol=1e-14)
    assert model.reward_class_masks.shape[0] == 16


def test_finite_horizon_mdp_base_case_is_immediate_reward():
    model = build_model()
    q_values = finite_horizon_mdp_q(model, horizon=2)

    np.testing.assert_array_equal(q_values[0], 0.0)
    np.testing.assert_allclose(q_values[1], model.rewards)
    assert np.all(q_values[2].max(axis=0) >= q_values[1].max(axis=0))


def test_more_observation_information_cannot_reduce_upper_bound():
    model = build_model()
    q_values = finite_horizon_mdp_q(model, horizon=8)
    belief = np.full((1, N_STATES), 1.0 / N_STATES)

    without_reward = observable_after_next_values(
        model,
        belief,
        q_values[7],
        previous_reward_observed=False,
    ).max()
    with_reward = observable_after_next_values(
        model,
        belief,
        q_values[7],
        previous_reward_observed=True,
    ).max()
    state_revealed_after_one = (belief @ q_values[8].T).max()

    assert with_reward >= without_reward
    assert state_revealed_after_one >= with_reward


def test_short_simulation_is_reproducible_and_accounts_for_every_action():
    model = build_model()
    horizon = 5
    episodes = 4
    q_values = finite_horizon_mdp_q(model, horizon=horizon)

    first = simulate_bayesian_policy(
        model,
        q_values,
        horizon=horizon,
        episodes=episodes,
        seed=7,
        previous_reward_observed=True,
    )
    second = simulate_bayesian_policy(
        model,
        q_values,
        horizon=horizon,
        episodes=episodes,
        seed=7,
        previous_reward_observed=True,
    )

    assert first == second
    assert sum(first["action_counts_total"].values()) == horizon * episodes
    assert np.isfinite(first["return_mean"])
