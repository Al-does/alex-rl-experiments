"""Focused recipe tests for the REINFORCE action-symmetry cycle."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_6.design import (
    CYCLE_6_TRANSITION_MATRIX,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    BASE_MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    TOTAL_ENV_STEPS,
    ReinforceTransformerModel,
)
from harness.context import RunContext
from harness.hardware import PROFILES


@pytest.fixture
def smoke_context(tmp_path):
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_reinforce_variants_build_fresh_monte_carlo_configs(
    smoke_context,
    variant,
):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_6."
        f"variant_{variant}.experiment"
    )

    first = module.build_config(smoke_context)
    second = module.build_config(smoke_context)
    spec = first.get_rl_module_spec()

    assert first is not second
    assert first.env_config["task"]["kwargs"]["variant"] == variant
    assert first.gamma == 0.99
    assert first.lambda_ == 1.0
    assert first.use_critic is False
    assert first.use_gae is False
    assert first.use_kl_loss is False
    assert first.vf_loss_coeff == 0.0
    assert first.entropy_coeff == 0.0
    assert first.num_epochs == 1
    assert first.minibatch_size is None
    assert first.batch_mode == "complete_episodes"
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert spec.module_class is ReinforceTransformerModel
    assert spec.model_config["d_model"] == 64
    assert spec.model_config["n_layers"] == 4
    assert spec.model_config["n_heads"] == 1
    assert spec.model_config["context_len"] == 10
    first.validate()

    environment = HMMEnv(first.env_config)
    try:
        assert environment.action_space.n == 3
        np.testing.assert_array_equal(
            environment.model.transition_matrix,
            CYCLE_6_TRANSITION_MATRIX,
        )
    finally:
        environment.close()


def test_cycle_6_budget_and_model_match_requested_recipe():
    assert TOTAL_ENV_STEPS == 2_500_000
    assert BASE_MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 10,
        "max_seq_len": 32,
    }


def test_reinforce_model_supplies_device_native_zero_baseline():
    embeddings = torch.randn(2, 7, 64)
    values = ReinforceTransformerModel.compute_values(
        object(),
        {},
        embeddings=embeddings,
    )

    assert values.shape == (2, 7)
    assert values.device == embeddings.device
    assert values.dtype == embeddings.dtype
    assert torch.count_nonzero(values) == 0
