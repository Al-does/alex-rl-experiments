from __future__ import annotations

import numpy as np

from envs.hmm import HMMEnv, factor_marginals
from experiments.two_factor_reward_state_PPO_cycle_2.process import (
    environment_config,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.control_analysis import (
    HistoryData,
    _decode_observations,
    empirical_random_histories,
    replay_beliefs,
    shuffled_histories,
    uniform_random_histories,
)


def _history_from_environment(length: int = 40) -> HistoryData:
    config = environment_config("reward_both")
    config["diagnostics"] = {"belief": True}
    env = HMMEnv(config)
    try:
        observation, info = env.reset(seed=17)
        observations = []
        beliefs = []
        steps = []
        for step in range(length):
            observations.append(observation.copy())
            beliefs.append(np.asarray(info["belief_current"]).copy())
            steps.append(step)
            observation, _, terminated, truncated, info = env.step(step % 9)
            assert not terminated
            assert not truncated
    finally:
        env.close()
    joint = np.asarray(beliefs)
    factors = np.stack(factor_marginals(joint, (3, 3)), axis=1)
    return HistoryData(
        observations=np.asarray(observations),
        beliefs=joint,
        factor_beliefs=factors,
        episode_ids=np.zeros(length, dtype=np.int64),
        episode_steps=np.asarray(steps, dtype=np.int64),
        mask=np.asarray(steps) >= 8,
        metadata={"n_envs": 1},
    )


def _two_episode_history() -> HistoryData:
    observations = np.zeros((8, 18), dtype=np.float32)
    tokens = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    observations[np.arange(8), tokens] = 1.0
    for row, action in enumerate((0, 1, 2, 3, 4, 5), start=1):
        if row == 4:
            continue
        observations[row, 9 + action] = 1.0
    observations[5, 9 + 4] = 1.0
    observations[6, 9 + 5] = 1.0
    observations[7, 9 + 6] = 1.0
    return HistoryData(
        observations=observations,
        beliefs=np.zeros((8, 9)),
        factor_beliefs=np.zeros((8, 2, 3)),
        episode_ids=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        episode_steps=np.asarray([0, 1, 2, 3, 0, 1, 2, 3]),
        mask=np.ones(8, dtype=bool),
        metadata={"n_envs": 2},
    )


def test_belief_replay_matches_public_environment_diagnostics() -> None:
    data = _history_from_environment()
    joint, factors = replay_beliefs(data, condition="reward_both")

    np.testing.assert_allclose(joint, data.beliefs, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        factors,
        data.factor_beliefs,
        atol=1e-12,
        rtol=0.0,
    )


def test_history_shuffle_is_episode_local_and_preserves_resets() -> None:
    data = _two_episode_history()
    shuffled = shuffled_histories(data, seed=23)

    np.testing.assert_array_equal(
        shuffled.observations[[0, 4]],
        data.observations[[0, 4]],
    )
    for indices in (np.arange(0, 4), np.arange(4, 8)):
        original_rows = sorted(map(bytes, data.observations[indices[1:]]))
        shuffled_rows = sorted(map(bytes, shuffled.observations[indices[1:]]))
        assert shuffled_rows == original_rows
    np.testing.assert_array_equal(
        shuffled_histories(data, seed=23).observations,
        shuffled.observations,
    )


def test_random_histories_are_deterministic_valid_discrete_inputs() -> None:
    data = _two_episode_history()
    empirical = empirical_random_histories(data, seed=31)
    uniform = uniform_random_histories(data, seed=37)

    np.testing.assert_array_equal(
        empirical_random_histories(data, seed=31).observations,
        empirical.observations,
    )
    np.testing.assert_array_equal(
        uniform_random_histories(data, seed=37).observations,
        uniform.observations,
    )
    for randomized in (empirical, uniform):
        tokens, actions = _decode_observations(randomized)
        assert np.all((0 <= tokens) & (tokens < 9))
        assert np.all(actions[randomized.episode_steps == 0] == -1)
        assert np.all(
            (0 <= actions[randomized.episode_steps > 0])
            & (actions[randomized.episode_steps > 0] < 9)
        )


def test_shuffled_original_and_recomputed_targets_are_distinct() -> None:
    data = _history_from_environment()
    shuffled = shuffled_histories(data, seed=41)
    joint, _ = replay_beliefs(shuffled, condition="reward_both")

    assert np.max(np.abs(joint - data.beliefs)) > 1e-3
