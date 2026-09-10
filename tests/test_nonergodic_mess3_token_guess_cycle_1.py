from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)
from experiments.nonergodic_mess3_token_guess_cycle_1 import analysis
from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_PARAMETERS,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    TOKEN_COUNT,
    environment_config,
    mess3_edge_matrices,
    nonergodic_mess3_model,
)
from experiments.nonergodic_mess3_token_guess_cycle_1.shared import (
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
    build_config,
    resolved_recipe,
)
from experiments.nonergodic_mess3_token_guess_cycle_1.task import (
    NextTokenGuessTask,
)
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


def _module() -> FactoredReproductionActorCritic:
    return FactoredReproductionActorCritic(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(TOKEN_COUNT,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(TOKEN_COUNT),
        model_config=dict(MODEL_CONFIG),
    ).eval()


def test_article_component_edge_matrices_and_nonergodic_composition():
    model = nonergodic_mess3_model()
    assert model.n_states == STATE_COUNT == 6
    assert model.n_tokens == TOKEN_COUNT == 3
    np.testing.assert_allclose(
        model.initial_distribution.reshape(2, 3).sum(axis=1),
        [0.5, 0.5],
    )
    for component, parameters in enumerate(COMPONENT_PARAMETERS):
        start = component * STATES_PER_COMPONENT
        stop = start + STATES_PER_COMPONENT
        np.testing.assert_allclose(
            model.edge_transition_matrices[:, start:stop, start:stop],
            mess3_edge_matrices(
                x=float(parameters["x"]),
                alpha=float(parameters["alpha"]),
            ),
        )
    np.testing.assert_array_equal(
        model.transition_matrix[:3, 3:],
        np.zeros((3, 3)),
    )
    np.testing.assert_array_equal(
        model.transition_matrix[3:, :3],
        np.zeros((3, 3)),
    )


def test_episode_chooses_one_component_and_token_guess_timing_is_delayed():
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
        assert isinstance(env.task, NextTokenGuessTask)
        np.testing.assert_array_equal(observation, np.zeros(TOKEN_COUNT))
        component = info["state_current"] // STATES_PER_COMPONENT
        pending_token = info["raw_token_current"]
        next_observation, reward, terminated, truncated, step_info = env.step(
            pending_token
        )
        assert reward == 1.0
        assert not terminated and not truncated
        assert step_info["raw_token_before"] == pending_token
        np.testing.assert_array_equal(
            next_observation,
            np.eye(TOKEN_COUNT, dtype=np.float32)[pending_token],
        )
        for _ in range(EPISODE_LENGTH - 1):
            _, _, _, truncated, step_info = env.step(0)
            assert step_info["state_current"] // STATES_PER_COMPONENT == component
        assert truncated
    finally:
        env.close()


def test_guesses_do_not_control_nonergodic_mess3_dynamics():
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
            [0, 1, 2] * 32,
            [2, 1, 0] * 32,
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


def test_fresh_gamma_zero_ppo_config_and_article_recipe(tmp_path):
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
    assert config.env_config["episode_length"] == EPISODE_LENGTH == 127
    assert config.rl_module_spec.module_class is FactoredReproductionActorCritic
    assert config.rl_module_spec.model_config["d_model"] == 128
    assert config.rl_module_spec.model_config["n_layers"] == 4
    assert config.rl_module_spec.model_config["n_heads"] == 4
    assert config.rl_module_spec.model_config["d_mlp"] == 512
    assert config.rl_module_spec.model_config["context_length"] == CONTEXT_LENGTH
    assert config.rl_module_spec.model_config["positional_embedding"] == "rope"
    recipe = resolved_recipe(context)
    assert recipe["objective"] == (
        "sampled next-token correctness only; no cross-entropy loss"
    )
    assert recipe["components"] == [dict(value) for value in COMPONENT_PARAMETERS]
    assert recipe["component_prior"] == [0.5, 0.5]
    assert recipe["previous_reward_in_observation"] is False
    assert recipe["previous_action_in_observation"] is False
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 1_024
    assert (
        resolved_recipe(replace(context, smoke=False))["total_env_steps"]
        == TOTAL_ENV_STEPS
        == 2_500_000
    )


def test_model_emits_one_logit_per_shared_mess3_token():
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


def test_probe_targets_separate_weighted_belief_from_next_token_distribution():
    data = analysis.collect_probe_data(
        _module(),
        n_steps=16,
        seed=np.random.SeedSequence(71),
        device=torch.device("cpu"),
    )
    assert data.activations.shape == (16, 4, 128)
    assert data.weighted_beliefs.shape == (16, STATE_COUNT)
    assert data.component_posteriors.shape == (16, 2)
    assert data.next_token_distributions.shape == (16, TOKEN_COUNT)
    np.testing.assert_allclose(data.weighted_beliefs.sum(axis=1), 1.0)
    np.testing.assert_allclose(data.component_posteriors.sum(axis=1), 1.0)
    np.testing.assert_allclose(data.next_token_distributions.sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        data.rewards,
        (data.actions == data.hidden_tokens).astype(np.float64),
    )


@pytest.mark.parametrize("action", [-1, 3, 1.5, True, "1"])
def test_task_rejects_invalid_token_guesses(action):
    env = HMMEnv(environment_config())
    try:
        env.reset(seed=3)
        with pytest.raises(ValueError, match="token guess"):
            env.step(action)
    finally:
        env.close()
