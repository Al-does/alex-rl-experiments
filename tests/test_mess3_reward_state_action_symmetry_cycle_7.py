"""Focused recipe tests for the ctx32 entropy-anneal action-symmetry cycle."""

from __future__ import annotations

import importlib

import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    BASE_MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    ReinforceTransformerModel,
    environment_config,
    write_budget_spec,
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


@pytest.mark.parametrize(
    ("variant", "condition", "temperature", "entropy_coeff"),
    (
        (1, "ctx32_ent", 1.0, [[0, 0.03], [1, 0.0]]),
        (2, "ctx32_ent", 1.0, [[0, 0.03], [1, 0.0]]),
        (3, "ctx32_ent", 1.0, [[0, 0.03], [1, 0.0]]),
    ),
)
def test_ctx32_arms_change_only_temperature_and_context(
    smoke_context, variant, condition, temperature, entropy_coeff
):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_7."
        f"variant_{variant}_{condition}.experiment"
    )

    config = module.build_config(smoke_context)
    spec = config.get_rl_module_spec()

    assert config.env_config["task"]["kwargs"]["variant"] == variant
    assert spec.module_class is ReinforceTransformerModel
    assert spec.model_config["sampling_temperature"] == temperature
    assert spec.model_config["context_len"] == 32
    assert spec.model_config["grad_checkpointing"] is True
    for key in ("d_model", "n_layers", "n_heads", "max_seq_len"):
        assert spec.model_config[key] == BASE_MODEL_CONFIG[key]
    assert config.lr == 4.2e-4
    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.gamma == 0.99
    assert config.lambda_ == 1.0
    assert config.entropy_coeff == entropy_coeff
    assert config.use_critic is False
    config.validate()


def test_ctx32_ent_schedule_reaches_zero_1m_before_target(tmp_path):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_7."
        "variant_2_ctx32_ent.experiment"
    )
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )
    write_budget_spec(context, 10_000_000)

    assert module.entropy_schedule(context) == [[0, 0.03], [9_000_000, 0.0]]


def test_ctx32_logits_divided_in_rollout_and_train(tmp_path):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_7."
        "variant_2_ctx32_ent.experiment"
    )
    environment = HMMEnv(environment_config(2))
    try:
        torch.manual_seed(11)
        policy = ReinforceTransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config=dict(module.CTX32_ENT_MODEL_CONFIG),
        ).eval()
    finally:
        environment.close()

    embeddings = torch.randn(2, 3, 64)
    expected = (
        policy.heads.action_distribution_inputs(embeddings) / 1.0
    )
    for training in (False, True):
        outputs = policy._outputs(embeddings, None, training=training)
        torch.testing.assert_close(
            outputs[Columns.ACTION_DIST_INPUTS], expected
        )
    torch.testing.assert_close(
        policy.action_distribution_inputs(embeddings), expected
    )

    batch = {
        Columns.OBS: torch.randn(1, 2, policy._obs_dim),
        Columns.STATE_IN: {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in policy.get_initial_state().items()
        },
    }
    train = policy.forward_train(batch)
    rollout = policy.forward_exploration(batch)
    torch.testing.assert_close(
        train[Columns.ACTION_DIST_INPUTS],
        rollout[Columns.ACTION_DIST_INPUTS],
    )
    assert policy.sequence_lookback == 128
