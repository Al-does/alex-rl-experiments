"""Scientific and wiring tests for MESS3 feedback cycle 1."""

from __future__ import annotations

import numpy as np
import pytest

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    make_transducer_target,
)
from experiments.mess3_feedback_cycle_1.analysis import _action_corruptor
from experiments.mess3_feedback_cycle_1.shared import (
    DEFAULT_SEEDS,
    ENV_CONFIG,
    ETA,
    TOTAL_ENV_STEPS,
    build_config,
)
from experiments.mess3_feedback_cycle_1.task import FeedbackNextTokenTask
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_feedback_matrices_are_symmetric_attractors():
    environment = HMMEnv(ENV_CONFIG)
    try:
        task = environment.task
        assert isinstance(task, FeedbackNextTokenTask)
        baseline = environment.model.transition_matrix
        for action in range(3):
            expected = (1.0 - ETA) * baseline
            expected = expected.copy()
            expected[:, action] += ETA
            actual = task.transition_matrix_for_action(action)
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_allclose(actual.sum(axis=1), 1.0)
            assert not actual.flags.writeable
        np.testing.assert_allclose(
            task.transition_matrix_for_action(1),
            np.roll(
                np.roll(task.transition_matrix_for_action(0), 1, axis=0),
                1,
                axis=1,
            ),
        )
    finally:
        environment.close()


def test_delay_one_observation_contains_token_and_previous_action():
    environment = HMMEnv(
        {
            **ENV_CONFIG,
            "diagnostics": {
                "belief": True,
                "tokens": True,
                "transitions": True,
            },
        }
    )
    try:
        observation, _ = environment.reset(seed=11)
        assert observation.shape == (6,)
        np.testing.assert_array_equal(observation[3:], np.zeros(3))
        next_observation, _, _, _, info = environment.step(2)
        np.testing.assert_array_equal(next_observation[3:], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            info["executed_transition_matrix"],
            environment.task.transition_matrix_for_action(2),
        )
    finally:
        environment.close()


def test_transducer_target_matches_action_conditioned_environment_belief():
    environment = HMMEnv(
        {
            **ENV_CONFIG,
            "diagnostics": {
                "belief": True,
                "tokens": True,
                "transitions": True,
            },
        }
    )
    try:
        initial, operator, initial_operator = make_transducer_target(environment)
        assert initial_operator is None
        _, info = environment.reset(seed=17)
        belief = initial
        for action in (0, 2, 1, 1, 0):
            _, _, _, _, info = environment.step(action)
            from analysis.probes import predictive_belief_update

            belief = predictive_belief_update(belief, operator(info))
            np.testing.assert_allclose(
                belief,
                info["belief_current"],
                atol=1e-12,
            )
    finally:
        environment.close()


def test_action_corruptions_touch_only_previous_action_block():
    observations = np.asarray(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    masked = _action_corruptor("mask")(observations)
    np.testing.assert_array_equal(masked[:, :3], observations[:, :3])
    np.testing.assert_array_equal(masked[:, 3:], 0.0)
    shuffled = _action_corruptor("shuffle", seed=4)(observations)
    np.testing.assert_array_equal(shuffled[:, :3], observations[:, :3])
    assert sorted(map(tuple, shuffled[:, 3:])) == sorted(
        map(tuple, observations[:, 3:])
    )
    np.testing.assert_array_equal(
        observations[0],
        [1, 0, 0, 1, 0, 0],
    )


def test_plain_ppo_recipe_is_locked(tmp_path):
    config = build_config(_context(tmp_path))
    assert ETA == pytest.approx(0.10)
    assert TOTAL_ENV_STEPS == 2_000_000
    assert DEFAULT_SEEDS == (42, 43, 44, 45, 46)
    assert config.gamma == 0.0
    assert config.lambda_ == 0.0
    assert config.env_config == ENV_CONFIG
    assert config.train_batch_size_per_learner == 2_048
    assert config.rl_module_spec.model_config["context_length"] == 10
    assert config.learner_config_dict == {}
