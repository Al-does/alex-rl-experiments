from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_analysis import (
    HistoryData,
    _decode_observations,
    empirical_random_histories,
    replay_beliefs,
    shuffled_histories,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import (
    _select_records,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    environment_config,
)


def _history_from_environment(length: int = 160) -> HistoryData:
    config = environment_config(3)
    config["randomize_first_episode_length"] = False
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
            observation, _, terminated, truncated, info = env.step(step % 3)
            assert not terminated
            assert not truncated
    finally:
        env.close()
    return HistoryData(
        observations=np.asarray(observations),
        activations=np.zeros((length, 4)),
        beliefs=np.asarray(beliefs),
        episode_ids=np.zeros(length, dtype=np.int64),
        episode_steps=np.asarray(steps, dtype=np.int64),
        mask=np.asarray(steps) >= 128,
        metadata={"n_envs": 1},
    )


def _two_episode_history() -> HistoryData:
    observations = np.zeros((8, 6), dtype=np.float32)
    tokens = np.asarray([0, 1, 2, 0, 1, 2, 0, 1])
    observations[np.arange(8), tokens] = 1.0
    for row, action in enumerate((0, 1, 2), start=1):
        observations[row, 3 + action] = 1.0
    for row, action in enumerate((1, 2, 0), start=5):
        observations[row, 3 + action] = 1.0
    return HistoryData(
        observations=observations,
        activations=np.zeros((8, 4)),
        beliefs=np.zeros((8, 3)),
        episode_ids=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        episode_steps=np.asarray([0, 1, 2, 3, 0, 1, 2, 3]),
        mask=np.ones(8, dtype=bool),
        metadata={"n_envs": 2},
    )


def test_belief_replay_matches_public_environment_diagnostics() -> None:
    data = _history_from_environment()

    np.testing.assert_allclose(
        replay_beliefs(data, variant=3),
        data.beliefs,
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


def test_random_histories_are_deterministic_valid_inputs() -> None:
    data = _two_episode_history()
    randomized = empirical_random_histories(data, seed=31)

    np.testing.assert_array_equal(
        empirical_random_histories(data, seed=31).observations,
        randomized.observations,
    )
    tokens, actions = _decode_observations(randomized)
    assert np.all((0 <= tokens) & (tokens < 3))
    assert np.all(actions[randomized.episode_steps == 0] == -1)
    assert np.all(
        (0 <= actions[randomized.episode_steps > 0])
        & (actions[randomized.episode_steps > 0] < 3)
    )


def test_checkpoint_selection_uses_shared_stages_and_final() -> None:
    path = next(
        Path(
            "experiments/mess3_reward_state_action_symmetry_cycle_7/"
            "variant_1_ctx32_ent/results"
        ).glob("*-seed55-10m/checkpoint_probe_curve.json")
    )

    selected = _select_records(path)

    assert [stage for stage, _ in selected] == [
        "init",
        "iteration_1",
        "iteration_2",
        "iteration_4",
        "iteration_8",
        "iteration_16",
        "final",
    ]
    assert [record.get("training_iteration") for _, record in selected] == [
        None,
        1,
        2,
        4,
        8,
        16,
        41,
    ]
