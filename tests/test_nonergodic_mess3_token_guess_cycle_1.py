from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.connectors.common import AddTimeDimToBatchAndZeroPad
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from torch import nn

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
    GatedGELUMLP,
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
    ALL_ONE_COMPONENT_BATCH_PROBABILITY,
    MINIBATCH_SIZE,
    MIN_EPISODES_PER_TRAIN_BATCH,
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
    assert config.rl_module_spec.model_config["max_seq_len"] == CONTEXT_LENGTH
    assert config.rl_module_spec.model_config["activation"] == "gated_gelu"
    assert config.rl_module_spec.model_config["normalization"] == "rms_norm"
    assert config.rl_module_spec.model_config["positional_embedding"] == "rope"
    assert config.rollout_fragment_length == "auto"
    assert config.batch_mode == "complete_episodes"
    assert config.env_runner_cls is FreshEpisodeSingleAgentEnvRunner
    recipe = resolved_recipe(context)
    assert recipe["objective"] == (
        "sampled next-token correctness only; no cross-entropy loss"
    )
    assert recipe["components"] == [dict(value) for value in COMPONENT_PARAMETERS]
    assert recipe["component_prior"] == [0.5, 0.5]
    assert recipe["previous_reward_in_observation"] is False
    assert recipe["previous_action_in_observation"] is False
    assert recipe["env_runner"] == (
        "harness.env_runners:FreshEpisodeSingleAgentEnvRunner"
    )
    assert recipe["environment_seed_semantics"] == (
        "the fixed worker seed initializes each vector environment once; "
        "later complete-episode resets advance the same RNG stream"
    )
    assert recipe["training_batch_component_mix"] == {
        "sampling": "independent equal-probability draw per complete episode",
        "minimum_episodes_per_full_train_batch": 259,
        "probability_full_train_batch_uses_one_component_only": (
            ALL_ONE_COMPONENT_BATCH_PROBABILITY
        ),
    }
    assert MIN_EPISODES_PER_TRAIN_BATCH == 259
    assert ALL_ONE_COMPONENT_BATCH_PROBABILITY < 3e-78
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS == 1_024
    assert (
        resolved_recipe(replace(context, smoke=False))["total_env_steps"]
        == TOTAL_ENV_STEPS
        == 10_000_000
    )
    assert (
        build_config(replace(context, smoke=False)).minibatch_size
        == MINIBATCH_SIZE
        == 1_024
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


def test_model_matches_article_rmsnorm_gated_gelu_and_bos_sequence():
    torch.manual_seed(37)
    module = _module()
    assert isinstance(module.encoder.final_norm, nn.RMSNorm)
    assert all(
        isinstance(block.attention_norm, nn.RMSNorm)
        and isinstance(block.mlp_norm, nn.RMSNorm)
        and isinstance(block.mlp, GatedGELUMLP)
        for block in module.encoder.blocks
    )

    block_inputs = []
    handle = module.encoder.blocks[0].register_forward_pre_hook(
        lambda _, inputs: block_inputs.append(inputs[0].detach().clone())
    )
    try:
        state = {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in module.get_initial_state().items()
        }
        _, state = module.encode_step(torch.zeros((1, TOKEN_COUNT)), state)
        bos_input = block_inputs.pop()
        torch.testing.assert_close(
            bos_input[0, -1],
            module.encoder.bos_embedding,
        )

        token = torch.eye(TOKEN_COUNT)[1].unsqueeze(0)
        module.encode_step(token, state)
        token_input = block_inputs.pop()
        torch.testing.assert_close(
            token_input[0, -2],
            module.encoder.bos_embedding,
        )
        torch.testing.assert_close(
            token_input[0, -1],
            module.encoder.input_embedding(token)[0],
        )
    finally:
        handle.remove()


def test_rllib_uses_128_step_bptt_without_splitting_complete_episodes():
    module = _module()
    connector = AddTimeDimToBatchAndZeroPad(as_learner_connector=True)
    for length, expected_seq_lens in ((EPISODE_LENGTH, [127]), (129, [128, 1])):
        observations = [np.zeros(TOKEN_COUNT, dtype=np.float32)]
        observations.extend(
            np.eye(TOKEN_COUNT, dtype=np.float32)[
                index % TOKEN_COUNT
            ]
            for index in range(length)
        )
        episode = SingleAgentEpisode(
            observations=observations,
            actions=[0] * length,
            rewards=[0.0] * length,
            len_lookback_buffer=0,
        )
        key = (episode.id_,)
        output = connector(
            rl_module=module,
            batch={Columns.OBS: {key: observations[:-1]}},
            episodes=[episode],
            shared_data={},
        )
        assert np.asarray(output[Columns.OBS][key]).shape == (
            len(expected_seq_lens),
            CONTEXT_LENGTH,
            TOKEN_COUNT,
        )
        assert output[Columns.SEQ_LENS][key] == expected_seq_lens
        assert int(np.asarray(output[Columns.LOSS_MASK][key]).sum()) == length


def test_full_training_batch_draws_many_independent_components():
    env = HMMEnv(
        {
            **environment_config(),
            "diagnostics": {"state": True},
        }
    )
    try:
        components = []
        _, info = env.reset(seed=42)
        components.append(info["state_current"] // STATES_PER_COMPONENT)
        for _ in range(MIN_EPISODES_PER_TRAIN_BATCH - 1):
            _, info = env.reset()
            components.append(info["state_current"] // STATES_PER_COMPONENT)
        assert set(components) == {0, 1}
        assert 90 < sum(components) < 170
    finally:
        env.close()


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


def test_probe_next_token_target_uses_source_belief_not_an_extra_transition():
    model = nonergodic_mess3_model()
    config = environment_config()
    config["diagnostics"] = {
        "belief": True,
        "state": True,
        "tokens": True,
    }
    env = HMMEnv(config)
    try:
        _, info = env.reset(seed=83)
        observation, _, _, _, info = env.step(info["raw_token_current"])
        targets = analysis._target_adapter(
            observation[None],
            [info],
            np.asarray([1]),
        )
        source = targets["weighted_belief"][0]
        pending = targets["next_token_distribution"][0]
        arrival = np.asarray(info["belief_current"])
        np.testing.assert_allclose(
            source @ model.transition_matrix,
            arrival,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            pending,
            source @ model.emission_matrix,
            atol=1e-12,
        )
        assert not np.allclose(
            pending,
            arrival @ model.emission_matrix,
        )
    finally:
        env.close()


@pytest.mark.parametrize("action", [-1, 3, 1.5, True, "1"])
def test_task_rejects_invalid_token_guesses(action):
    env = HMMEnv(environment_config())
    try:
        env.reset(seed=3)
        with pytest.raises(ValueError, match="token guess"):
            env.step(action)
    finally:
        env.close()
