"""Scientific and architecture tests for both cycle-2 algorithms."""

from __future__ import annotations

import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_5.task import (
    ActionSymmetryTask,
)
from experiments.two_factor_reward_state_PPO_cycle_2.model import TwoFactorRewardPPO
from experiments.two_factor_reward_state_PPO_cycle_2.shared import (
    ENTROPY_COEFF,
    MODEL_CONFIG as PPO_MODEL_CONFIG,
    SMOKE_BATCH_SIZE as PPO_SMOKE_BATCH_SIZE,
    TOTAL_ENV_STEPS as PPO_TOTAL_ENV_STEPS,
)
from experiments.two_factor_reward_state_SAC_cycle_2.design import (
    analytic_design_summary,
)
from experiments.two_factor_reward_state_SAC_cycle_2.model import (
    FLAT_OBSERVATION_WIDTH,
    TwoFactorRewardSAC,
    TwoFactorRoPESACEncoder,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    JOINT_TOKEN_COUNT,
    LOCAL_CONTEXT_LENGTH,
    MESS3_ALPHA,
    SAC_HISTORY_LENGTH,
    TRANSFORMER_LOOKBACK,
    controlled_factor_model,
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.shared import (
    MODEL_CONFIG as SAC_MODEL_CONFIG,
    SMOKE_BATCH_SIZE as SAC_SMOKE_BATCH_SIZE,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import (
    ACTION_PAIRS,
    CONDITIONS,
    N_ACTIONS,
    factor_transition,
    joint_transition,
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


def test_each_factor_exactly_matches_action_symmetry_variant_3():
    reference = ActionSymmetryTask(
        model=controlled_factor_model(),
        variant=3,
        effect_size=1.5,
    )
    for action in range(3):
        np.testing.assert_allclose(
            factor_transition(action),
            reference.transition_matrix_for_action(action),
        )
    summary = analytic_design_summary()
    assert summary["oracle_policy_by_state"] == [1, 2, 0]
    assert summary["oracle_single_factor_state_2_occupancy"] == pytest.approx(
        0.356518584849733,
        abs=1e-12,
    )


def test_product_actions_control_factors_independently():
    assert CONDITIONS == ("reward_both", "reward_factor_1")
    assert len(ACTION_PAIRS) == N_ACTIONS == 9
    for action, (first, second) in enumerate(ACTION_PAIRS):
        np.testing.assert_allclose(
            joint_transition(action),
            np.kron(factor_transition(first), factor_transition(second)),
        )


def test_environment_uses_sticky_high_information_process():
    environment = HMMEnv(environment_config("reward_both"))
    try:
        observation, _ = environment.reset(seed=9)
        assert MESS3_ALPHA == 0.85
        assert environment.model.n_states == 9
        assert environment.model.n_tokens == JOINT_TOKEN_COUNT == 9
        assert environment.action_space == gym.spaces.Discrete(9)
        assert observation.shape == (FLAT_OBSERVATION_WIDTH,)
        assert SAC_HISTORY_LENGTH == TRANSFORMER_LOOKBACK + 1 == 41
    finally:
        environment.close()


@pytest.mark.parametrize("algorithm", ["SAC", "PPO"])
@pytest.mark.parametrize("condition", CONDITIONS)
def test_each_cycle_2_leaf_builds_a_fresh_recipe(tmp_path, algorithm, condition):
    module = importlib.import_module(
        f"experiments.two_factor_reward_state_{algorithm}_cycle_2."
        f"{condition}.experiment"
    )
    first = module.build_config(_context(tmp_path))
    second = module.build_config(_context(tmp_path))
    assert first is not second
    assert first.seed == 42
    assert first.num_env_runners == 0
    assert first.env_config["task"]["kwargs"]["condition"] == condition
    if algorithm == "SAC":
        assert first.train_batch_size_per_learner == SAC_SMOKE_BATCH_SIZE
        assert first.rl_module_spec.module_class is TwoFactorRewardSAC
    else:
        assert first.train_batch_size_per_learner == PPO_SMOKE_BATCH_SIZE
        assert first.rl_module_spec.module_class is TwoFactorRewardPPO
        assert first.entropy_coeff == ENTROPY_COEFF == 0.003
        assert PPO_TOTAL_ENV_STEPS == 5_000_000


def test_models_match_cycle_5_rope_transformer_contract():
    for config in (SAC_MODEL_CONFIG, PPO_MODEL_CONFIG):
        assert config["d_model"] == 64
        assert config["n_layers"] == 4
        assert config["n_heads"] == 1
        assert config["context_len"] == LOCAL_CONTEXT_LENGTH == 10

    sac = TwoFactorRewardSAC(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(FLAT_OBSERVATION_WIDTH,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(N_ACTIONS),
        model_config={**SAC_MODEL_CONFIG, "twin_q": True},
        inference_only=False,
    )
    sac.make_target_networks()
    assert isinstance(sac.pi_encoder, TwoFactorRoPESACEncoder)
    assert isinstance(sac.qf_encoder, TwoFactorRoPESACEncoder)
    assert sac.pi_encoder.encoder.lookback == TRANSFORMER_LOOKBACK
    observations = torch.zeros((2, FLAT_OBSERVATION_WIDTH))
    observations[:, :JOINT_TOKEN_COUNT] = torch.eye(JOINT_TOKEN_COUNT)[:2]
    assert sac.actor_hidden(observations).shape == (2, 64)

    ppo = TwoFactorRewardPPO(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(JOINT_TOKEN_COUNT + N_ACTIONS,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(N_ACTIONS),
        model_config=PPO_MODEL_CONFIG,
    )
    assert ppo.encoder.lookback == TRANSFORMER_LOOKBACK
    assert ppo.encoder.n_heads == 1
