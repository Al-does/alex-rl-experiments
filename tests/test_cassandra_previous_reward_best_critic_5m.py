"""Tests for previous-reward rerun with best critic hyperparameters."""

from __future__ import annotations

import pytest

from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_previous_reward_best_critic_5m.experiment import (
    build_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    CassandraPreviousRewardObservationEnv,
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


def test_previous_reward_best_critic_uses_visible_reward_and_best_vf(smoke_context):
    config = build_config(smoke_context)

    assert config.env is CassandraPreviousRewardObservationEnv
    assert config.env_config["action_scope"] == "targeted"
    assert config.env_config["initial_state_distribution"] == "all_good"
    assert config.gamma == pytest.approx(0.990)
    assert config.lambda_ == pytest.approx(0.95)
    assert config.vf_clip_param == pytest.approx(100.0)
    assert config.vf_loss_coeff == pytest.approx(0.01)
    assert config.entropy_coeff == pytest.approx(0.03)
    assert config.rl_module_spec.model_config["context_len"] == 256


def test_previous_reward_best_critic_rejects_mismatched_seed(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=7,
        smoke=True,
        hardware=PROFILES["cpu"],
    )

    with pytest.raises(ValueError, match="require seed 42"):
        build_config(context)
