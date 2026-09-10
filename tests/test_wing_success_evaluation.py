from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv
from experiments.wing_token_guess_cycle_1.process import environment_config as passive_config
from experiments.wing_two_factor_explore_cycle_1.control_data import collect_control_data, replay_beliefs
from experiments.wing_two_factor_explore_cycle_1.evaluate_success import episode_statistics, evaluate_module_success, policy_success_values
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
from experiments.wing_two_factor_explore_cycle_2.process import environment_config as controlled_config


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def _module(config):
    env = HMMEnv(config)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(12)
            return WingActorCritic(
                observation_space=env.observation_space, action_space=env.action_space,
                model_config={"d_model": 12, "n_layers": 1, "n_heads": 1, "d_mlp": 24, "context_length": 8, "sampling_temperature": 1.5, "positional_embedding": "rope"},
            )
    finally:
        env.close()


@pytest.mark.parametrize("condition", ["token_guess", "reward_both", "reward_factor_1"])
def test_complete_episode_evaluation_is_deterministic_and_includes_reset(condition):
    config = passive_config() if condition == "token_guess" else controlled_config(condition)
    module = _module(config)
    first = evaluate_module_success(module, env_config=config, condition=condition, n_episodes=4, horizon=9, n_envs=2, seed=12)
    second = evaluate_module_success(module, env_config=config, condition=condition, n_episodes=4, horizon=9, n_envs=2, seed=12)
    assert first == second
    assert first["n_decisions"] == 36
    assert first["n_episodes"] == 4
    assert first["warmup"] == 0
    assert first["environment_config"]["randomize_first_episode_length"] is False
    assert first["ci95"][0] <= first["mean"] <= first["ci95"][1]
    assert 0 < first["mean"] < 1
    assert all(not block._forward_hooks for block in module.encoder.blocks)


def test_token_optimal_success_is_joint_max_not_mean_factor_accuracy():
    config = passive_config()
    config.update(episode_length=8, randomize_first_episode_length=False)
    data = collect_control_data(_module(config), env_config=config, n_steps=16, n_envs=2, warmup=0, seed=13, device=torch.device("cpu"))
    parameters = {"alpha": 0.94, "x": 0.4, "strength": None}
    _, marginal = replay_beliefs(data, **parameters)
    joint = (marginal[:, 0, :, None] * marginal[:, 1, None, :]).reshape(-1, 4)
    optimal = np.eye(4)[joint.argmax(axis=1)]
    actual = policy_success_values(replace(data, policy_probabilities=optimal), parameters=parameters, condition="token_guess")
    np.testing.assert_allclose(actual, marginal.max(axis=-1).prod(axis=1))
    assert not np.allclose(actual, marginal.max(axis=-1).mean(axis=1))
    uniform = policy_success_values(replace(data, policy_probabilities=np.full((16, 4), 0.25)), parameters=parameters, condition="token_guess")
    np.testing.assert_allclose(uniform, 0.25)


def test_uniform_rotation_actions_have_one_third_arrival_occupancy():
    config = controlled_config("reward_both")
    config.update(episode_length=8, randomize_first_episode_length=False)
    data = collect_control_data(_module(config), env_config=config, n_steps=16, n_envs=2, warmup=0, seed=14, device=torch.device("cpu"))
    data = replace(data, policy_probabilities=np.full((16, 9), 1 / 9))
    for condition in ("reward_both", "reward_factor_1"):
        actual = policy_success_values(data, parameters={"alpha": 0.94, "x": 0.4, "strength": 1.0}, condition=condition)
        np.testing.assert_allclose(actual, 1 / 3)


def test_episode_statistics_do_not_treat_steps_as_independent():
    result = episode_statistics(np.repeat([0.2, 0.8], 100), np.repeat([0, 1], 100), 100)
    assert result["mean"] == pytest.approx(0.5)
    assert result["standard_error"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="complete"):
        episode_statistics([0.2, 0.2, 0.8], [0, 0, 1], 2)
    with pytest.raises(ValueError, match="two independent"):
        episode_statistics([0.2, 0.3], [0, 0], 2)
