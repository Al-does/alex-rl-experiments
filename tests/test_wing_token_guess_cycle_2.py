from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.wing_token_guess_cycle_2 import analysis
from experiments.wing_token_guess_cycle_2.process import (
    CONTEXT_LENGTH,
    FACTOR_COUNT,
    TOKEN_COUNT,
    environment_config,
)
from experiments.wing_token_guess_cycle_2.shared import (
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
    build_config,
    resolved_recipe,
)
from experiments.wing_token_guess_cycle_2.task import WingTokenGuessTask
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
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


def _module() -> WingActorCritic:
    return WingActorCritic(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(TOKEN_COUNT,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(TOKEN_COUNT),
        model_config=dict(MODEL_CONFIG),
    ).eval()


def test_delay_one_action_logits_guess_pending_single_wing_token():
    config = environment_config()
    config["diagnostics"] = {
        "belief": True,
        "tokens": True,
        "transitions": True,
    }
    env = HMMEnv(config)
    try:
        observation, info = env.reset(seed=5)
        assert config["model"]["factory"] == "envs.wing.model:wing_model"
        assert env.model.n_states == 3
        assert env.model.n_tokens == TOKEN_COUNT == 2
        assert env.action_space == gym.spaces.Discrete(TOKEN_COUNT)
        assert isinstance(env.task, WingTokenGuessTask)
        np.testing.assert_array_equal(
            observation,
            np.zeros(TOKEN_COUNT),
        )
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
            np.eye(TOKEN_COUNT, dtype=np.float32)[pending_token],
        )
    finally:
        env.close()


def test_guesses_do_not_control_wing_dynamics():
    first = HMMEnv(
        {
            **environment_config(),
            "diagnostics": {"state": True, "tokens": True},
        }
    )
    second = HMMEnv(
        {
            **environment_config(),
            "diagnostics": {"state": True, "tokens": True},
        }
    )
    try:
        first.reset(seed=17)
        second.reset(seed=17)
        for first_guess, second_guess in zip(
            [0, 1] * 32,
            [1, 0] * 32,
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


def test_previous_reward_is_not_in_observation_or_belief():
    config = {
        **environment_config(),
        "diagnostics": {
            "belief": True,
            "state": True,
            "tokens": True,
        },
    }
    normal = HMMEnv(config)
    changed_reward = HMMEnv(config)
    changed_reward.task.reward = lambda event, decision: (
        7.0,
        {"replacement_reward": 7.0},
    )
    try:
        normal_observation, normal_info = normal.reset(seed=29)
        changed_observation, changed_info = changed_reward.reset(seed=29)
        np.testing.assert_array_equal(normal_observation, changed_observation)
        assert normal_observation.shape == (TOKEN_COUNT,)
        assert normal.config.observation.action is None
        assert not normal.config.observation.belief
        assert not normal.config.observation.hidden_state
        for guess in np.random.default_rng(8).integers(
            TOKEN_COUNT,
            size=64,
        ):
            normal_step = normal.step(int(guess))
            changed_step = changed_reward.step(int(guess))
            np.testing.assert_array_equal(normal_step[0], changed_step[0])
            np.testing.assert_array_equal(
                normal_step[4]["belief_current"],
                changed_step[4]["belief_current"],
            )
            assert normal_step[4]["state_current"] == changed_step[4]["state_current"]
            assert normal_step[4]["raw_token_current"] == changed_step[4]["raw_token_current"]
            assert changed_step[1] == 7.0
            normal_info, changed_info = normal_step[4], changed_step[4]
        np.testing.assert_array_equal(
            normal_info["belief_current"],
            changed_info["belief_current"],
        )
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
    assert config.env_config["observation"] == {
        "token": {"depth": 1},
        "action": None,
    }
    assert config.rl_module_spec.module_class is WingActorCritic
    assert config.rl_module_spec.model_config["context_length"] == CONTEXT_LENGTH
    assert config.rl_module_spec.model_config["positional_embedding"] == "rope"
    assert config.rl_module_spec.model_config["sampling_temperature"] == 1.0
    recipe = resolved_recipe(context)
    assert recipe["study"] == "wing_token_guess_cycle_2"
    assert recipe["environment"]["model"] == {
        "factory": "envs.wing.model:wing_model",
        "kwargs": {"alpha": 0.94, "x": 0.4},
    }
    assert recipe["changes_from_cycle_1"]["token_count"] == {
        "before": 4,
        "after": 2,
    }
    assert recipe["previous_reward_in_observation"] is False
    assert recipe["previous_action_in_observation"] is False
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 4_096
    assert (
        resolved_recipe(replace(context, smoke=False))["total_env_steps"]
        == TOTAL_ENV_STEPS
        == 2_500_000
    )


def test_model_emits_one_logit_per_wing_token():
    torch.manual_seed(31)
    module = _module()
    state = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in module.get_initial_state().items()
    }
    batch = {
        Columns.OBS: torch.zeros((1, 2, TOKEN_COUNT)),
        Columns.STATE_IN: state,
    }
    outputs = module.forward_train(batch)
    assert outputs[Columns.ACTION_DIST_INPUTS].shape == (
        1,
        2,
        TOKEN_COUNT,
    )
    assert module.compute_values(batch).shape == (1, 2)


def test_probe_collection_uses_reward_aligned_targets_without_reward_inputs():
    data = analysis.collect_probe_data(
        _module(),
        n_steps=64,
        seed=np.random.SeedSequence(71),
        device=torch.device("cpu"),
    )
    assert data.activations.shape == (64, 3, 64)
    assert FACTOR_COUNT == 1
    assert data.joint_beliefs.shape == (64, 3)
    assert data.factor_beliefs.shape == (64, 1, 3)
    assert data.observations.shape == (64, TOKEN_COUNT)
    assert data.episode_steps.min() >= CONTEXT_LENGTH
    assert data.product_consistency_max_abs < 1e-10
    np.testing.assert_array_equal(
        data.rewards,
        (data.actions == data.hidden_tokens).astype(np.float64),
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
