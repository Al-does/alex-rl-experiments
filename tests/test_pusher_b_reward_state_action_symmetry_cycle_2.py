from __future__ import annotations

from dataclasses import replace
import importlib

import numpy as np
import pytest

from envs.hmm import HMMEnv
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.design import (
    MAX_CONTROLLED_TRANSITION_PROBABILITY,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.task import (
    N_ACTIONS,
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
    ActionSymmetryTask,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_2.design import (
    analytic_design_summary,
    condition_design_summary,
    controlled_kernels,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_2.process import (
    DESTINATION_EMISSION_MATRIX,
    EFFECT_SIZE,
    EPISODE_LENGTH,
    REWARD_STATE,
    TRANSITION_MATRIX,
    environment_config,
    sticky_cycle_edge_matrices,
    sticky_cycle_model,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_2.shared import (
    MINIBATCH_SIZE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
    TRAIN_BATCH_SIZE,
    build_config,
    resolved_recipe,
)
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
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


def _diagnostic_config(variant: int) -> dict[str, object]:
    config = environment_config(variant)
    config["diagnostics"] = {
        "belief": True,
        "state": True,
        "tokens": True,
        "transitions": True,
    }
    return config


def test_sticky_cycle_edge_matrices_match_the_specification():
    expected = np.asarray(
        [
            [
                [0.450, 0.045, 0.010],
                [0.020, 0.675, 0.015],
                [0.030, 0.030, 0.225],
            ],
            [
                [0.450, 0.015, 0.030],
                [0.020, 0.225, 0.045],
                [0.030, 0.010, 0.675],
            ],
        ]
    )
    np.testing.assert_allclose(sticky_cycle_edge_matrices(), expected)


def test_sticky_cycle_model_is_normalized_and_stationary():
    model = sticky_cycle_model()
    np.testing.assert_allclose(
        model.transition_matrix,
        TRANSITION_MATRIX,
    )
    np.testing.assert_allclose(
        model.edge_transition_matrices.sum(axis=0),
        TRANSITION_MATRIX,
    )
    np.testing.assert_allclose(
        model.emission_matrix,
        TRANSITION_MATRIX @ DESTINATION_EMISSION_MATRIX,
    )
    np.testing.assert_allclose(
        model.initial_distribution,
        np.full(3, 1.0 / 3.0),
    )
    np.testing.assert_allclose(
        model.initial_distribution @ model.transition_matrix,
        model.initial_distribution,
    )


@pytest.mark.parametrize("variant", [2, 3])
def test_controlled_edges_preserve_destination_token_conditionals(variant):
    model = sticky_cycle_model()
    task = ActionSymmetryTask(
        model=model,
        variant=variant,
        reward_state=REWARD_STATE,
        effect_size=EFFECT_SIZE,
    )
    np.testing.assert_allclose(
        task.transition_matrix_for_action(NOOP_ACTION),
        model.transition_matrix,
    )
    for action in range(N_ACTIONS):
        transition = task.transition_matrix_for_action(action)
        edges = task.edge_transition_matrices_for_action(action)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)
        np.testing.assert_allclose(edges.sum(axis=0), transition)
        np.testing.assert_allclose(
            edges / transition[None, :, :],
            np.broadcast_to(
                DESTINATION_EMISSION_MATRIX.T[:, None, :],
                edges.shape,
            ),
        )


@pytest.mark.parametrize(
    ("variant", "expected_policy", "distinct_actions"),
    [
        (2, (NOOP_ACTION, POSITIVE_ACTION, POSITIVE_ACTION), 2),
        (3, (NOOP_ACTION, POSITIVE_ACTION, NEGATIVE_ACTION), 3),
    ],
)
def test_full_state_oracle_has_requested_action_symmetry(
    variant,
    expected_policy,
    distinct_actions,
):
    summary = condition_design_summary(variant)
    assert tuple(summary["oracle_policy"]) == expected_policy
    assert len(set(summary["oracle_policy"])) == distinct_actions
    assert summary["oracle_gap"] == pytest.approx(0.20034155129918663)
    assert summary["maximum_controlled_transition_probability"] == (
        pytest.approx(0.952456443456355)
    )
    assert summary["maximum_controlled_transition_probability"] < (
        MAX_CONTROLLED_TRANSITION_PROBABILITY
    )
    assert summary["minimum_controlled_transition_probability"] > 0.0
    assert summary["oracle_stationary_reward_occupancy"] == pytest.approx(
        0.7895008047802166
    )


def test_design_summary_contains_only_variants_two_and_three():
    summary = analytic_design_summary()
    assert EFFECT_SIZE == 2.5
    assert REWARD_STATE == "A"
    assert [condition["variant"] for condition in summary["conditions"]] == [
        2,
        3,
    ]


def test_bos_and_delayed_token_action_observation():
    env = HMMEnv(_diagnostic_config(3))
    try:
        observation, info = env.reset(seed=23)
        np.testing.assert_array_equal(
            observation,
            np.zeros(2 + N_ACTIONS, dtype=np.float32),
        )
        pending_token = int(info["raw_token_current"])
        observation, _, _, _, info = env.step(NEGATIVE_ACTION)
        np.testing.assert_array_equal(
            observation[:2],
            np.eye(2, dtype=np.float32)[pending_token],
        )
        np.testing.assert_array_equal(
            observation[2:],
            np.eye(N_ACTIONS, dtype=np.float32)[NEGATIVE_ACTION],
        )
        assert info["visible_source_token"] == pending_token
    finally:
        env.close()


def test_reward_uses_pre_transition_state_a():
    env = HMMEnv(_diagnostic_config(2))
    try:
        env.reset(seed=31)
        for action in [0, 1, 2] * 20:
            _, reward, _, _, info = env.step(action)
            assert reward == float(info["state_before"] == 0)
    finally:
        env.close()


def test_complete_episode_ppo_config_and_recipe(tmp_path):
    context = _context(tmp_path)
    config = build_config(context, 3)
    assert config is not build_config(context, 3)
    assert config.seed == 42
    assert config.gamma == 0.99
    assert config.lambda_ == 0.95
    assert config.clip_param == 0.2
    assert config.use_critic and config.use_gae
    assert not config.use_kl_loss
    assert config.vf_loss_coeff == 0.5
    assert config.entropy_coeff == 0.01
    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert config.num_epochs == 6
    assert config.num_env_runners == 0
    assert config.env_config == environment_config(3)
    assert config.env_config["delay"] == 1
    assert config.env_config["episode_length"] == EPISODE_LENGTH == 127
    assert config.env_config["randomize_first_episode_length"] is False
    assert config.env_runner_cls is FreshEpisodeSingleAgentEnvRunner
    assert config.rollout_fragment_length == "auto"
    assert config.batch_mode == "complete_episodes"
    assert config.rl_module_spec.model_config == MODEL_CONFIG
    assert MODEL_CONFIG["context_length"] == 128
    assert MODEL_CONFIG["max_seq_len"] == 128
    assert MODEL_CONFIG["training_sequence_mode"] == "complete_episode"

    recipe = resolved_recipe(context, 3)
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 1_024
    assert recipe["effect_size"] == EFFECT_SIZE
    assert recipe["previous_action_in_observation"] is True
    assert recipe["environment"] == environment_config(3)
    full_context = replace(context, smoke=False)
    full_config = build_config(full_context, 3)
    assert full_config.train_batch_size_per_learner == TRAIN_BATCH_SIZE
    assert full_config.minibatch_size == MINIBATCH_SIZE
    assert resolved_recipe(full_context, 3)["total_env_steps"] == TOTAL_ENV_STEPS
    assert TOTAL_ENV_STEPS == 50_000_000


@pytest.mark.parametrize("variant", [2, 3])
def test_leaf_modules_are_importable_and_bound_correctly(tmp_path, variant):
    module = importlib.import_module(
        "experiments.pusher_b_reward_state_action_symmetry_cycle_2."
        f"variant_{variant}.experiment"
    )
    assert callable(module.run)
    config = module.build_config(_context(tmp_path))
    assert config.env_config == environment_config(variant)


def test_controlled_kernel_shapes():
    transitions, edges = controlled_kernels(3)
    assert transitions.shape == (N_ACTIONS, 3, 3)
    assert edges.shape == (N_ACTIONS, 2, 3, 3)


@pytest.mark.parametrize("variant", [1, 4])
def test_environment_config_rejects_unknown_variants(variant):
    with pytest.raises(ValueError, match="variant"):
        environment_config(variant)
