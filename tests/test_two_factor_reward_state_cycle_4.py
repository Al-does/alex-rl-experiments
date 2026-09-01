"""Recipe tests for baseline-free two-factor REINFORCE cycle 4."""

from __future__ import annotations

import importlib

import pytest
import torch

from envs.hmm import HMMEnv
from experiments.two_factor_reward_state_PPO_cycle_2.task import CONDITIONS
from experiments.two_factor_reward_state_REINFORCE_cycle_3.shared import (
    build_config as build_cycle_3_config,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.model import (
    TwoFactorRewardReinforceCycle4,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    LEARNING_RATE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    TOTAL_ENV_STEPS,
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


@pytest.mark.parametrize("condition", CONDITIONS)
def test_cycle_4_preserves_task_and_builds_simple_reinforce_recipe(
    tmp_path,
    condition,
):
    module = importlib.import_module(
        "experiments.two_factor_reward_state_REINFORCE_cycle_4."
        f"{condition}.experiment"
    )
    first = module.build_config(_context(tmp_path))
    second = module.build_config(_context(tmp_path))
    cycle_3 = build_cycle_3_config(_context(tmp_path), condition)
    spec = first.get_rl_module_spec()

    assert first is not second
    assert first.env_config == cycle_3.env_config
    assert first.env_config["task"]["kwargs"]["condition"] == condition
    assert first.seed == 42
    assert first.num_env_runners == 0
    assert first.lr == LEARNING_RATE == 4.2e-4
    assert first.gamma == 0.99
    assert first.lambda_ == 1.0
    assert first.use_critic is False
    assert first.use_gae is False
    assert first.use_kl_loss is False
    assert first.vf_loss_coeff == 0.0
    assert first.entropy_coeff == 0.0
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size is None
    assert first.num_epochs == 1
    assert first.batch_mode == "complete_episodes"
    assert spec.module_class is TwoFactorRewardReinforceCycle4
    first.validate()

    environment = HMMEnv(first.env_config)
    try:
        assert environment.observation_space.shape == (18,)
        assert environment.action_space.n == 9
    finally:
        environment.close()


def test_cycle_4_preserves_architecture_and_budget():
    assert TOTAL_ENV_STEPS == 8_000_000
    assert MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 10,
        "max_seq_len": 32,
    }


def test_cycle_4_value_api_is_an_inert_device_native_zero_baseline():
    embeddings = torch.randn(3, 5, 64)
    values = TwoFactorRewardReinforceCycle4.compute_values(
        object(),
        {},
        embeddings=embeddings,
    )

    assert values.shape == (3, 5)
    assert values.device == embeddings.device
    assert values.dtype == embeddings.dtype
    assert torch.count_nonzero(values) == 0
