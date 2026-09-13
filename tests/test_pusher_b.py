from __future__ import annotations

from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)
from experiments.pusher_b.process import (
    BOS_TOKEN,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    PRESETS,
    STATE_COUNT,
    TOKEN_COUNT,
    environment_config,
    pusher_b_edge_matrices,
    pusher_b_model,
)
from experiments.pusher_b.rl import (
    ENTROPY_COEFF_SCHEDULE,
    LEARNING_RATE_SCHEDULE,
    MINIBATCH_SIZE,
    MODEL_CONFIG as RL_MODEL_CONFIG,
    NUM_ENVS_PER_ENV_RUNNER,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
    TRAIN_BATCH_SIZE,
    build_config,
    resolved_recipe,
)
from experiments.pusher_b.supervised import (
    CHECKPOINT_STEPS,
    MODEL_CONFIG as SUPERVISED_MODEL_CONFIG,
    SequenceSampler,
    TrainingConfig,
    beliefs_from_tokens,
    next_token_distributions,
)
from experiments.pusher_b.supervised_model import PusherBTransformer
from experiments.pusher_b.task import NextTokenGuessTask
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


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (
            "b90",
            np.asarray(
                [
                    [
                        [0.025, 0.45, 0.025],
                        [0.045, 0.045, 0.81],
                        [0.09, 0.005, 0.005],
                    ],
                    [
                        [0.025, 0.025, 0.45],
                        [0.09, 0.005, 0.005],
                        [0.045, 0.81, 0.045],
                    ],
                ]
            ),
        ),
        (
            "b10",
            np.asarray(
                [
                    [
                        [0.025, 0.45, 0.025],
                        [0.005, 0.005, 0.09],
                        [0.81, 0.045, 0.045],
                    ],
                    [
                        [0.025, 0.025, 0.45],
                        [0.81, 0.045, 0.045],
                        [0.005, 0.09, 0.005],
                    ],
                ]
            ),
        ),
    ],
)
def test_edge_matrices_match_requested_presets(preset, expected):
    parameters = PRESETS[preset]
    edges = pusher_b_edge_matrices(**parameters)
    np.testing.assert_allclose(edges, expected)

    model = pusher_b_model(preset)
    assert model.n_states == STATE_COUNT == 3
    assert model.n_tokens == TOKEN_COUNT == 2
    assert model.state_labels == ("A", "B", "C")
    assert model.token_labels == ("0", "1")
    np.testing.assert_allclose(model.edge_transition_matrices, expected)
    np.testing.assert_allclose(model.transition_matrix, expected.sum(axis=0))
    np.testing.assert_allclose(
        model.emission_matrix,
        expected.sum(axis=2).T,
    )
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.emission_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        model.initial_distribution @ model.transition_matrix,
        model.initial_distribution,
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_delayed_token_guess_environment_timing(preset):
    config = environment_config(preset)
    config["diagnostics"] = {
        "state": True,
        "tokens": True,
        "transitions": True,
    }
    env = HMMEnv(config)
    try:
        observation, info = env.reset(seed=5)
        assert isinstance(env.task, NextTokenGuessTask)
        np.testing.assert_array_equal(observation, np.zeros(TOKEN_COUNT))
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
            _, _, _, truncated, _ = env.step(0)
        assert truncated
    finally:
        env.close()


@pytest.mark.parametrize("preset", PRESETS)
def test_actions_do_not_control_pusher_b_dynamics(preset):
    config = {
        **environment_config(preset),
        "diagnostics": {"state": True, "tokens": True},
    }
    first = HMMEnv(config)
    second = HMMEnv(config)
    try:
        first.reset(seed=17)
        second.reset(seed=17)
        for first_guess, second_guess in zip([0, 1] * 20, [1, 0] * 20):
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


@pytest.mark.parametrize("preset", PRESETS)
def test_supervised_sampler_beliefs_and_model(preset):
    tokens = SequenceSampler(preset=preset, seed=11).sample(
        8,
        emission_count=9,
    )
    assert tokens.shape == (8, 10)
    np.testing.assert_array_equal(tokens[:, 0], BOS_TOKEN)
    assert set(np.unique(tokens[:, 1:])).issubset({0, 1})
    beliefs = beliefs_from_tokens(tokens, preset=preset)
    distributions = next_token_distributions(beliefs, preset=preset)
    assert beliefs.shape == (8, 10, STATE_COUNT)
    assert distributions.shape == (8, 10, TOKEN_COUNT)
    np.testing.assert_allclose(beliefs.sum(axis=-1), 1.0)
    np.testing.assert_allclose(distributions.sum(axis=-1), 1.0)

    model = PusherBTransformer()
    logits = model(torch.from_numpy(tokens))
    assert logits.shape == (8, 10, TOKEN_COUNT + 1)
    assert model.config == SUPERVISED_MODEL_CONFIG
    assert model.config.d_model == 128
    assert model.config.n_layers == 4
    assert model.config.n_heads == 4
    assert model.config.d_mlp == 512
    assert model.config.context_length == CONTEXT_LENGTH


def test_supervised_recipe_matches_pr_130_guide():
    config = TrainingConfig()
    assert config.total_steps == CHECKPOINT_STEPS[-1] == 10_000
    assert config.batch_size == 512
    assert config.learning_rate == 1e-3
    assert (config.beta1, config.beta2) == (0.9, 0.999)
    assert config.weight_decay == 0.0
    smoke = TrainingConfig.smoke()
    assert smoke.total_steps == 2
    assert smoke.checkpoint_steps == (1, 2)


@pytest.mark.parametrize("preset", PRESETS)
def test_fresh_ppo_config_matches_pr_127_guide(tmp_path, preset):
    context = _context(tmp_path)
    config = build_config(context, preset=preset)
    assert config is not build_config(context, preset=preset)
    assert config.seed == 42
    assert config.gamma == 0.0
    assert config.lambda_ == 0.0
    assert config.clip_param == 0.2
    assert config.use_critic and config.use_gae
    assert not config.use_kl_loss
    assert config.vf_loss_coeff == 0.25
    assert config.entropy_coeff == ENTROPY_COEFF_SCHEDULE
    assert config.lr == LEARNING_RATE_SCHEDULE
    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert config.num_epochs == 6
    assert config.num_env_runners == 0
    assert config.num_envs_per_env_runner == 1
    assert config.env_config == environment_config(preset)
    assert config.rl_module_spec.module_class is FactoredReproductionActorCritic
    assert config.rl_module_spec.model_config == RL_MODEL_CONFIG
    assert config.rollout_fragment_length == "auto"
    assert config.batch_mode == "complete_episodes"
    assert config.env_runner_cls is FreshEpisodeSingleAgentEnvRunner

    recipe = resolved_recipe(context, preset=preset)
    assert recipe["parameters"] == PRESETS[preset]
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 1_024
    assert recipe["objective"] == (
        "sampled next-token correctness only; no cross-entropy loss"
    )
    full_context = replace(context, smoke=False)
    full_config = build_config(full_context, preset=preset)
    assert full_config.train_batch_size_per_learner == TRAIN_BATCH_SIZE == 262_144
    assert full_config.minibatch_size == MINIBATCH_SIZE == 8_192
    assert full_config.num_envs_per_env_runner == NUM_ENVS_PER_ENV_RUNNER == 19
    assert (
        resolved_recipe(full_context, preset=preset)["total_env_steps"]
        == TOTAL_ENV_STEPS
        == 15_000_000
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "experiments.pusher_b.supervised_b90.experiment",
        "experiments.pusher_b.supervised_b10.experiment",
        "experiments.pusher_b.rl_b90.experiment",
        "experiments.pusher_b.rl_b10.experiment",
    ],
)
def test_all_four_experiment_leaves_import(module_name):
    module = importlib.import_module(module_name)
    assert callable(module.run)
