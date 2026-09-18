"""Scientific and wiring tests for the RRXOR PPO reproduction."""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.rrxor_token_guess_cycle_1 import analysis
from experiments.rrxor_token_guess_cycle_1.model import (
    MODEL_CONFIG,
    RRXORActorCritic,
)
from experiments.rrxor_token_guess_cycle_1.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    RRXOR_STATIONARY,
    RRXOR_TENSOR,
    environment_config,
    filter_source_belief,
    next_token_probabilities,
    reachable_beliefs,
    rrxor_model,
)
from experiments.rrxor_token_guess_cycle_1.shared import (
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
    _headline,
    build_config,
    resolved_recipe,
)
from experiments.rrxor_token_guess_cycle_1.task import RRXORTokenGuessTask
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


def _module() -> RRXORActorCritic:
    return RRXORActorCritic(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(2,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(2),
        model_config=dict(MODEL_CONFIG),
    ).eval()


def test_rrxor_tensor_stationarity_and_xor_semantics():
    model = rrxor_model()
    assert RRXOR_TENSOR.shape == (2, 5, 5)
    np.testing.assert_allclose(
        RRXOR_TENSOR.sum(axis=0),
        model.transition_matrix,
    )
    np.testing.assert_allclose(
        RRXOR_STATIONARY @ model.transition_matrix,
        RRXOR_STATIONARY,
    )
    assert RRXOR_TENSOR[0, 0, 1] == 0.5
    assert RRXOR_TENSOR[1, 0, 2] == 0.5
    assert RRXOR_TENSOR[0, 1, 4] == 0.5
    assert RRXOR_TENSOR[1, 1, 3] == 0.5
    assert RRXOR_TENSOR[0, 2, 3] == 0.5
    assert RRXOR_TENSOR[1, 2, 4] == 0.5
    assert RRXOR_TENSOR[1, 3, 0] == 1.0
    assert RRXOR_TENSOR[0, 4, 0] == 1.0
    np.testing.assert_allclose(
        next_token_probabilities(RRXOR_STATIONARY),
        [0.5, 0.5],
    )


def test_rrxor_has_36_exact_reachable_belief_states():
    beliefs = reachable_beliefs()
    assert beliefs.shape == (36, 5)
    np.testing.assert_allclose(beliefs.sum(axis=1), 1.0)
    assert np.all(beliefs >= 0.0)


def test_delay_one_observation_scores_the_pending_edge_token():
    config = environment_config()
    config["diagnostics"] = {
        "belief": True,
        "state": True,
        "tokens": True,
        "transitions": True,
    }
    env = HMMEnv(config)
    try:
        observation, info = env.reset(seed=5)
        assert env.model.n_states == 5
        assert env.model.n_tokens == 2
        assert isinstance(env.task, RRXORTokenGuessTask)
        np.testing.assert_array_equal(observation, np.zeros(2))
        assert info["visible_source_token"] is None
        pending_token = info["raw_token_current"]
        next_observation, reward, terminated, truncated, step_info = env.step(
            pending_token
        )
        assert reward == 1.0
        assert not terminated and not truncated
        assert step_info["raw_token_before"] == pending_token
        assert step_info["visible_source_token"] == pending_token
        np.testing.assert_array_equal(
            next_observation,
            np.eye(2, dtype=np.float32)[pending_token],
        )
    finally:
        env.close()


def test_guesses_do_not_change_rrxor_dynamics():
    config = {
        **environment_config(),
        "diagnostics": {"state": True, "tokens": True},
    }
    first = HMMEnv(config)
    second = HMMEnv(config)
    try:
        first.reset(seed=17)
        second.reset(seed=17)
        for first_guess, second_guess in zip(
            [0, 1] * 5,
            [1, 0] * 5,
        ):
            first_step = first.step(first_guess)
            second_step = second.step(second_guess)
            assert (
                first_step[4]["state_current"],
                first_step[4]["raw_token_current"],
            ) == (
                second_step[4]["state_current"],
                second_step[4]["raw_token_current"],
            )
    finally:
        first.close()
        second.close()


def test_reward_action_and_hidden_state_do_not_leak_into_observation():
    config = {
        **environment_config(),
        "diagnostics": {"belief": True, "state": True, "tokens": True},
    }
    normal = HMMEnv(config)
    changed_reward = HMMEnv(config)
    changed_reward.task.reward = lambda event, decision: (
        7.0,
        {"replacement_reward": 7.0},
    )
    try:
        normal_observation, _ = normal.reset(seed=29)
        changed_observation, _ = changed_reward.reset(seed=29)
        np.testing.assert_array_equal(normal_observation, changed_observation)
        assert normal_observation.shape == (2,)
        assert normal.config.observation.action is None
        assert not normal.config.observation.belief
        assert not normal.config.observation.hidden_state
        for guess in [0, 1] * 5:
            normal_step = normal.step(guess)
            changed_step = changed_reward.step(guess)
            np.testing.assert_array_equal(normal_step[0], changed_step[0])
            np.testing.assert_array_equal(
                normal_step[4]["belief_current"],
                changed_step[4]["belief_current"],
            )
            assert (
                normal_step[4]["state_current"]
                == changed_step[4]["state_current"]
            )
            assert (
                normal_step[4]["raw_token_current"]
                == changed_step[4]["raw_token_current"]
            )
            assert changed_step[1] == 7.0
    finally:
        normal.close()
        changed_reward.close()


def test_fresh_gamma_zero_ppo_config_and_smoke_recipe(tmp_path):
    context = _context(tmp_path)
    config = build_config(context)
    assert config is not build_config(context)
    assert config.seed == 42
    assert config.gamma == 0.0
    assert config.lambda_ == 0.0
    assert config.clip_param == 0.2
    assert config.use_critic and config.use_gae
    assert not config.use_kl_loss
    assert config.vf_loss_coeff == 0.5
    assert config.entropy_coeff == 0.0
    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert config.num_epochs == 6
    assert config.num_env_runners == 0
    assert config.env_config == environment_config()
    assert config.env_config["delay"] == 1
    assert config.rl_module_spec.module_class is RRXORActorCritic
    assert config.rl_module_spec.model_config == MODEL_CONFIG
    recipe = resolved_recipe(context)
    assert recipe["study"] == "rrxor_token_guess_cycle_1"
    assert recipe["process"]["reachable_belief_states"] == 36
    assert recipe["leakage"] == {
        "previous_reward": False,
        "previous_action": False,
        "belief": False,
        "hidden_state": False,
    }
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 4_096
    assert (
        resolved_recipe(replace(context, smoke=False))["total_env_steps"]
        == TOTAL_ENV_STEPS
        == 2_500_000
    )


def test_build_config_rejects_unresolved_seed(tmp_path):
    with pytest.raises(ValueError, match="resolved seed"):
        build_config(replace(_context(tmp_path), seed=None))


def test_paper_model_emits_two_logits_and_a_scalar_value():
    torch.manual_seed(31)
    module = _module()
    state = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in module.get_initial_state().items()
    }
    batch = {
        Columns.OBS: torch.zeros((1, 2, 2)),
        Columns.STATE_IN: state,
    }
    outputs = module.forward_train(batch)
    assert outputs[Columns.ACTION_DIST_INPUTS].shape == (1, 2, 2)
    assert module.compute_values(batch).shape == (1, 2)


def test_probe_headline_reads_shared_geometry_report():
    metrics = {
        "mse": 0.1,
        "r_squared": 0.2,
        "contrasts": {"next_token_invisible_1": {"r_squared": 0.3}},
    }
    report = {
        "agent_steps": 4_096,
        "training_iteration": 2,
        "policy": {
            "token_accuracy": 0.5,
            "bayes_expected_accuracy": 0.625,
        },
        "geometry": {
            "representations": {
                "layer_1": {"metrics": metrics},
                "concatenated_layers": {"metrics": metrics},
                "post_final_layer_norm": {"metrics": metrics},
            }
        },
        "coverage": {"test_beliefs_observed": 36},
    }
    headline = _headline(report)
    assert headline["belief_r2_per_layer"] == {"layer_1": 0.2}
    assert headline["concatenated_belief_r2"] == 0.2
    assert headline["concatenated_prediction_null_r2"] == {
        "next_token_invisible_1": 0.3
    }


def test_probe_collection_aligns_exact_beliefs_layers_and_rewards():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = analysis.collect_probe_data(
            _module(),
            n_steps=128,
            seed=np.random.SeedSequence(71),
            device=torch.device("cpu"),
        )
    finally:
        torch.set_num_threads(original_threads)
    assert data.layers.shape == (128, 4, 64)
    assert data.layers.reshape(128, -1).shape == (128, 256)
    assert data.post_final_norm.shape == (128, 64)
    assert data.policy_probabilities.shape == (128, 2)
    assert data.beliefs.shape == (128, 5)
    assert data.observations.shape == (128, 2)
    assert data.episode_steps.min() == 0
    assert data.episode_steps.max() == CONTEXT_LENGTH
    assert len(np.unique(data.episode_ids)) >= 16
    assert data.diagnostic_alignment_max_abs < 1e-10
    np.testing.assert_array_equal(
        data.rewards,
        (data.actions == data.hidden_tokens).astype(np.float64),
    )
    np.testing.assert_allclose(
        data.policy_probabilities.sum(axis=1),
        1.0,
        atol=1e-7,
    )

    for episode_id in np.unique(data.episode_ids):
        indices = np.flatnonzero(data.episode_ids == episode_id)
        order = indices[np.argsort(data.episode_steps[indices])]
        belief = RRXOR_STATIONARY.copy()
        for index in order:
            step = data.episode_steps[index]
            if step > 0:
                belief = filter_source_belief(
                    belief,
                    int(data.visible_tokens[index]),
                )
            np.testing.assert_allclose(data.beliefs[index], belief)
            np.testing.assert_allclose(
                data.next_token_probabilities[index],
                next_token_probabilities(belief),
            )


@pytest.mark.parametrize("action", [-1, 2, 1.5, True, "1"])
def test_task_rejects_invalid_token_guesses(action):
    env = HMMEnv(environment_config())
    try:
        env.reset(seed=3)
        with pytest.raises(ValueError, match="token guess"):
            env.step(action)
    finally:
        env.close()


def test_episode_and_context_lengths_preserve_paper_prefixes():
    assert CONTEXT_LENGTH == 10
    assert EPISODE_LENGTH == 11
