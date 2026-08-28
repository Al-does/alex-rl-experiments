"""Scientific, architecture, and wiring tests for two-factor reward-state PPO."""

from __future__ import annotations

import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv
from experiments.two_factor_reward_state_PPO_cycle_1.design import (
    REFERENCE_VALUES,
    fully_observed_occupancy,
)
from experiments.two_factor_reward_state_PPO_cycle_1.model import TwoFactorRewardPPO
from experiments.two_factor_reward_state_PPO_cycle_1.process import (
    CONTEXT_LENGTH,
    JOINT_TOKEN_COUNT,
    MESS3_ALPHA,
    environment_config,
)
from experiments.two_factor_reward_state_PPO_cycle_1.shared import (
    GAE_LAMBDA,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
)
from experiments.two_factor_reward_state_PPO_cycle_1.task import (
    ACTION_PAIRS,
    CONDITIONS,
    N_ACTIONS,
    joint_transition,
)
from harness.context import RunContext
from harness.hardware import PROFILES


OBSERVATION_WIDTH = JOINT_TOKEN_COUNT + N_ACTIONS


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def _module() -> TwoFactorRewardPPO:
    return TwoFactorRewardPPO(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(OBSERVATION_WIDTH,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(N_ACTIONS),
        model_config=MODEL_CONFIG,
    )


def test_scientific_process_and_action_contract_match_pr_65():
    assert MESS3_ALPHA == 0.55
    assert ACTION_PAIRS == (
        (0, 0),
        (1, 1),
        (1, 2),
        (0, 1),
        (0, 2),
        (2, 1),
        (2, 2),
        (1, 0),
        (2, 0),
    )
    assert fully_observed_occupancy() == pytest.approx(
        REFERENCE_VALUES["fully_observed"],
        abs=1e-12,
    )


def test_environment_emits_one_aligned_token_action_frame():
    config = environment_config("reward_both")
    config["diagnostics"] = {"transitions": True}
    environment = HMMEnv(config)
    try:
        observation, _ = environment.reset(seed=9)
        assert environment.observation_space.shape == (OBSERVATION_WIDTH,)
        assert observation[:JOINT_TOKEN_COUNT].sum() == 1.0
        assert observation[JOINT_TOKEN_COUNT:].sum() == 0.0

        next_observation, _, _, _, info = environment.step(2)
        np.testing.assert_allclose(
            info["executed_transition_matrix"],
            joint_transition(2),
        )
        assert next_observation[:JOINT_TOKEN_COUNT].sum() == 1.0
        assert next_observation[JOINT_TOKEN_COUNT + 2] == 1.0
        assert next_observation[JOINT_TOKEN_COUNT:].sum() == 1.0
    finally:
        environment.close()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_each_leaf_builds_a_fresh_twenty_million_step_ppo_recipe(
    tmp_path,
    condition,
):
    module = importlib.import_module(
        "experiments.two_factor_reward_state_PPO_cycle_1."
        f"{condition}.experiment"
    )
    context = _context(tmp_path)
    first = module.build_config(context)
    second = module.build_config(context)

    assert first is not second
    assert first.seed == 42
    assert first.gamma == 0.99
    assert first.lambda_ == GAE_LAMBDA == 0.95
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert first.num_env_runners == 0
    assert first.env_config["task"]["kwargs"]["condition"] == condition
    assert first.rl_module_spec.module_class is TwoFactorRewardPPO
    assert first.rl_module_spec.model_config["d_model"] == 64
    assert TOTAL_ENV_STEPS == 20_000_000


def test_actor_critic_uses_action_aware_64d_causal_transformer():
    torch.manual_seed(3)
    module = _module().eval()
    assert module.reproduction_config.d_model == 64
    assert module.reproduction_config.n_layers == 4
    assert module.reproduction_config.n_heads == 4
    assert module.reproduction_config.context_length == CONTEXT_LENGTH
    assert module.encoder.input_embedding.in_features == OBSERVATION_WIDTH

    first = torch.zeros((1, 4, OBSERVATION_WIDTH))
    first[0, :, 0] = 1.0
    second = first.clone()
    second[0, 2:, JOINT_TOKEN_COUNT + 2] = 1.0
    state = module.get_initial_state()
    context = torch.from_numpy(state["ctx"]).unsqueeze(0)
    lengths = torch.from_numpy(state["len"])
    first_residual = module.encode_chunks_pre_final_norm(context, lengths, first)
    second_residual = module.encode_chunks_pre_final_norm(context, lengths, second)

    torch.testing.assert_close(first_residual[:, :2], second_residual[:, :2])
    assert first_residual.shape == (1, 4, 64)
    assert not torch.allclose(first_residual[:, 2:], second_residual[:, 2:])
