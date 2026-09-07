from __future__ import annotations

import importlib
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
from experiments.wing_two_factor_explore_cycle_1.process import CONDITIONS, environment_config
from experiments.wing_two_factor_explore_cycle_1.shared import (
    MODEL_CONFIG,
    SAMPLING_TEMPERATURE,
    SMOKE_BATCH_SIZE,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def make_module():
    return WingActorCritic(
        observation_space=gym.spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
        action_space=gym.spaces.Discrete(9),
        model_config=dict(MODEL_CONFIG),
    ).eval()


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("reward_state", range(3))
def test_six_fresh_ppo_recipes(tmp_path, condition, reward_state):
    leaf = importlib.import_module(
        f"experiments.wing_two_factor_explore_cycle_1.{condition}_state_{reward_state}.experiment"
    )
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    config = leaf.build_config(context)
    assert config is not leaf.build_config(context)
    assert config.seed == 42
    assert config.use_critic and config.use_gae
    assert config.lambda_ == 0.95
    assert config.gamma == 0.99
    assert config.clip_param == 0.2
    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert config.num_env_runners == 0
    assert config.rl_module_spec.module_class is WingActorCritic
    assert config.env_config == environment_config(condition, reward_state)
    assert leaf.build_config(replace(context, smoke=False)).num_epochs == 6
    from experiments.wing_two_factor_explore_cycle_1.shared import resolved_recipe

    expected_budget = 50_000_000 if condition == "reward_both" else 30_000_000
    assert TOTAL_ENV_STEPS[condition] == expected_budget
    assert resolved_recipe(replace(context, smoke=False), condition, reward_state)["total_env_steps"] == expected_budget
    assert resolved_recipe(context, condition, reward_state)["total_env_steps"] == 2048
    assert config.rl_module_spec.model_config["positional_embedding"] == "rope"


@pytest.mark.parametrize("reward_state", range(3))
def test_reward_never_enters_observations_or_beliefs(reward_state):
    environments = []
    for condition in CONDITIONS:
        config = environment_config(condition, reward_state)
        config["diagnostics"] = {"belief": True, "state": True}
        environments.append(HMMEnv(config))
    try:
        observations = [env.reset(seed=73)[0] for env in environments]
        np.testing.assert_array_equal(*observations)
        assert observations[0].shape == (10,)
        assert observations[0][:4].sum() == 1
        assert observations[0][4:].sum() == 0
        rewards_differ = False
        for action in np.random.default_rng(15).integers(9, size=128):
            first, second = [env.step(int(action)) for env in environments]
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[4]["belief_current"], second[4]["belief_current"])
            np.testing.assert_array_equal(
                first[0][4:], np.eye(3)[list(divmod(int(action), 3))].reshape(-1)
            )
            assert first[0][4:].sum() == 2
            first_state, second_state = divmod(first[4]["state_current"], 3)
            assert first[1] == ((first_state == reward_state) + (second_state == reward_state)) / 2
            assert second[1] == float(first_state == reward_state)
            rewards_differ |= first[1] != second[1]
        assert rewards_differ
    finally:
        for env in environments:
            env.close()


def test_temperature_matches_rollout_and_ppo_likelihoods():
    torch.manual_seed(11)
    module = make_module()
    embeddings = torch.randn(2, 3, 64)
    expected = module.heads.action_distribution_inputs(embeddings) / SAMPLING_TEMPERATURE
    for training in (False, True):
        outputs = module._outputs(embeddings, None, training=training)
        torch.testing.assert_close(outputs[Columns.ACTION_DIST_INPUTS], expected)
    torch.testing.assert_close(module.action_distribution_inputs(embeddings), expected)
    batch = {
        Columns.OBS: torch.randn(1, 2, 10),
        Columns.STATE_IN: {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in module.get_initial_state().items()
        },
    }
    train = module.forward_train(batch)
    rollout = module.forward_exploration(batch)
    torch.testing.assert_close(train[Columns.ACTION_DIST_INPUTS], rollout[Columns.ACTION_DIST_INPUTS])
    assert module.compute_values(batch).shape == (1, 2)
    loss = train[Columns.ACTION_DIST_INPUTS].square().mean() + module.compute_values(batch).square().mean()
    loss.backward()
    assert module.encoder.input_embedding.weight.grad.abs().sum() > 0


def test_exact_32_frame_context_causality_and_action_sensitivity():
    torch.manual_seed(19)
    module = make_module()
    assert module.sequence_lookback == 31
    assert len(module.encoder.blocks) == 3
    assert module.reproduction_config.n_heads == 1
    state = module.get_initial_state()
    context = torch.from_numpy(state["ctx"]).unsqueeze(0)
    lengths = torch.from_numpy(state["len"])
    original = torch.zeros(1, 34, 10)
    original[:, :, 0] = 1
    changed = original.clone()
    changed[:, 0, 4] = 1
    with torch.no_grad():
        first = module.encode_chunks_pre_final_norm(context, lengths, original)
        second = module.encode_chunks_pre_final_norm(context, lengths, changed)
    assert first.shape == (1, 34, 64)
    assert not torch.allclose(first[:, 0], second[:, 0])
    torch.testing.assert_close(first[:, 32:], second[:, 32:])
    changed = original.clone()
    changed[:, 20:, 5] = 1
    with torch.no_grad():
        second = module.encode_chunks_pre_final_norm(context, lengths, changed)
    torch.testing.assert_close(first[:, :20], second[:, :20])
    assert not torch.allclose(first[:, 20:], second[:, 20:])


@pytest.mark.parametrize("state", [-1, 3, True])
def test_invalid_reward_state(state):
    with pytest.raises(ValueError):
        environment_config("reward_both", state)


def test_rope_attention_matches_explicit_relative_rotations():
    module = make_module()
    assert module.reproduction_config.positional_embedding == "rope"
    assert module.encoder.position_embedding is None
    assert not any("position_embedding" in name for name, _ in module.named_parameters())
    attention = module.encoder.blocks[0].attention
    inputs = torch.randn(2, 5, 64)
    allowed = torch.ones(5, 5, dtype=torch.bool).tril().expand(2, -1, -1)
    query, key, value = attention.qkv(inputs).reshape(2, 5, 3, 1, 64).permute(2, 0, 3, 1, 4).unbind(0)
    angles = torch.arange(5)[:, None] / (10000.0 ** (torch.arange(32) / 32))

    def rotate(tensor):
        pairs = tensor.reshape(2, 1, 5, 32, 2)
        even, odd = pairs.unbind(-1)
        return torch.stack((even * angles.cos() - odd * angles.sin(), even * angles.sin() + odd * angles.cos()), dim=-1).flatten(-2)

    scores = rotate(query) @ rotate(key).transpose(-1, -2) / 8
    expected = (scores.masked_fill(~allowed[:, None], -torch.inf).softmax(-1) @ value).transpose(1, 2).reshape(2, 5, 64)
    torch.testing.assert_close(attention(inputs, allowed), attention.output(expected))


def test_rope_strict_window_chunk_matches_incremental_steps():
    module = make_module()
    observations = torch.randn(2, 36, 10)
    state = {key: torch.from_numpy(value).unsqueeze(0).repeat(2, *([1] * value.ndim)) for key, value in module.get_initial_state().items()}
    with torch.no_grad():
        chunk = module.encode_chunks_pre_final_norm(state["ctx"], state["len"], observations)
        steps = []
        for observation in observations.unbind(1):
            residual, state = module.encode_step_pre_final_norm(observation, state)
            steps.append(residual)
    torch.testing.assert_close(chunk, torch.stack(steps, dim=1), atol=1e-6, rtol=1e-5)


def test_position_config_keeps_absolute_default_and_validates_rope():
    from experiments.factored_representations_reproduction_PPO_2026_08.model import FactoredReproductionModelConfig

    assert FactoredReproductionModelConfig().positional_embedding == "learned_absolute"
    with pytest.raises(ValueError):
        FactoredReproductionModelConfig(positional_embedding="unknown")
    with pytest.raises(ValueError):
        FactoredReproductionModelConfig(d_model=63, n_heads=1, positional_embedding="rope")
