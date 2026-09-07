"""Focused tests for the Cycle-6 independent token-flip diagnostic."""

from __future__ import annotations

import importlib

import numpy as np

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.analysis import (
    _coarse_spec,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.independent_flip_diagnostic.analysis import (
    _coarse_observations,
    _filter_targets,
    independently_flip_state_0_1_tokens,
    paired_local_flip_replay,
)
from learners.models import TransformerModel


def _observations(tokens: list[int]) -> np.ndarray:
    observations = np.zeros((len(tokens), 6), dtype=np.float32)
    observations[np.arange(len(tokens)), tokens] = 1.0
    observations[:, 3] = 1.0
    return observations


def _probe_data(observations: np.ndarray) -> ProbeData:
    length = len(observations)
    return ProbeData(
        activations=np.zeros((length, 1)),
        beliefs=np.zeros((length, 3)),
        diagnostic_beliefs=np.zeros((length, 3)),
        tokens=observations[:, :3].argmax(axis=1),
        previous_tokens=np.asarray([-1, *observations[:-1, :3].argmax(axis=1)]),
        env_indices=np.zeros(length, dtype=np.int64),
        episode_steps=np.arange(length, dtype=np.int64),
        states=np.zeros(length, dtype=np.int64),
        actions=np.zeros((length, 1)),
        rewards=np.zeros(length),
        observations=observations,
    )


def test_independent_flip_preserves_coarse_history_and_action_features():
    observations = _observations([0, 0, 1, 2, 1, 0, 2])

    randomized = independently_flip_state_0_1_tokens(
        observations,
        rng=np.random.default_rng(17),
        probability=0.5,
    )

    np.testing.assert_array_equal(randomized[:, 2], observations[:, 2])
    np.testing.assert_array_equal(randomized[:, 3:], observations[:, 3:])
    np.testing.assert_array_equal(
        randomized[:, :2].sum(axis=1),
        observations[:, :2].sum(axis=1),
    )
    assert np.any(randomized[:, :2] != observations[:, :2])
    assert np.any(randomized[:, :2] == observations[:, :2])


def test_independent_flip_probability_one_is_the_global_swap():
    observations = _observations([0, 1, 2, 0])

    randomized = independently_flip_state_0_1_tokens(
        observations,
        rng=np.random.default_rng(3),
        probability=1.0,
    )

    np.testing.assert_array_equal(
        randomized[:, :3].argmax(axis=1),
        [1, 0, 2, 1],
    )


def test_local_flip_changes_fine_target_but_not_coarse_target():
    environment = HMMEnv(
        importlib.import_module(
            "experiments.mess3_reward_state_action_symmetry_cycle_6.shared"
        ).environment_config(2)
    )
    try:
        initial = np.asarray(environment.model.initial_distribution)
        emission = np.asarray(environment.model.emission_matrix)
        transitions = {
            action: np.asarray(
                environment.task.transition_matrix_for_action(action)
            )
            for action in range(environment.action_space.n)
        }
        coarse_initial, coarse_emission, coarse_transitions = _coarse_spec(
            environment
        )
    finally:
        environment.close()
    factual = _observations([0, 0, 0])
    randomized = _observations([0, 1, 0])
    data = _probe_data(factual)

    factual_fine = _filter_targets(
        data,
        factual,
        initial=initial,
        emission=emission,
        transitions=transitions,
    )
    randomized_fine = _filter_targets(
        data,
        randomized,
        initial=initial,
        emission=emission,
        transitions=transitions,
    )
    factual_coarse = _filter_targets(
        data,
        _coarse_observations(factual),
        initial=coarse_initial,
        emission=coarse_emission,
        transitions=coarse_transitions,
    )
    randomized_coarse = _filter_targets(
        data,
        _coarse_observations(randomized),
        initial=coarse_initial,
        emission=coarse_emission,
        transitions=coarse_transitions,
    )

    assert abs(factual_fine[-1, 2] - randomized_fine[-1, 2]) > 1e-3
    np.testing.assert_allclose(randomized_coarse, factual_coarse, atol=1e-15)


def test_paired_local_flip_replay_reconstructs_factual_activations():
    config = {
        **importlib.import_module(
            "experiments.mess3_reward_state_action_symmetry_cycle_6.shared"
        ).environment_config(2),
        "episode_length": 128,
        "randomize_first_episode_length": False,
        "diagnostics": {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        },
    }

    def make_environment():
        return HMMEnv(config)

    environment = make_environment()
    try:
        target = make_transducer_target(environment)
        module = TransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config={
                "context_len": 2,
                "d_model": 24,
                "n_layers": 1,
                "n_heads": 3,
                "max_seq_len": 8,
            },
        )
    finally:
        environment.close()
    data = collect_probe_data(
        module,
        make_environment,
        n_steps=256,
        seed=42,
        policy_mode="random",
        n_envs=2,
        warmup=0,
        store_observations=True,
        initial_belief=target[0],
        action_outcome_operator=target[1],
        initial_outcome_operator=target[2],
    )

    replay = paired_local_flip_replay(
        module,
        data,
        randomization_seeds=[7, 8],
        device="cpu",
        warmup=4,
    )

    indices = np.asarray(replay["indices"])
    randomized = np.asarray(replay["randomized_observations"])
    assert len(indices) > 0
    assert float(replay["reconstruction_error"]) < 2e-5
    np.testing.assert_allclose(
        replay["factual_activations"],
        data.activations[indices],
        atol=2e-5,
    )
    np.testing.assert_array_equal(
        randomized[..., 2],
        np.broadcast_to(data.observations[..., 2], randomized[..., 2].shape),
    )
    assert np.sqrt(
        np.mean(
            np.square(
                np.asarray(replay["randomized_activations"])
                - np.asarray(replay["factual_activations"])[None, ...]
            )
        )
    ) > 0.0


def test_cycle_six_independent_flip_experiment_wiring():
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_6."
        "independent_flip_diagnostic.experiment"
    )

    assert module.CYCLE == 6
    assert module.VARIANT == 2
    assert callable(module.run)
