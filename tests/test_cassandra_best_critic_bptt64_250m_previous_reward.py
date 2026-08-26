"""Tests for the 250M targeted previous-reward best-critic BPTT-64 recipe."""

from __future__ import annotations

import pytest

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    BEST_VF_CLIP_PARAM,
    BEST_VF_LOSS_COEFF,
    TOTAL_ENV_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted_previous_reward.experiment import (
    ACTION_SCOPE,
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


def test_previous_reward_recipe_matches_targeted_bptt64_best_critic(smoke_context):
    first = build_config(smoke_context)
    second = build_config(smoke_context)

    assert first is not second
    assert first.seed == second.seed == 42
    assert first.env is CassandraPreviousRewardObservationEnv
    assert first.env_config["action_scope"] == ACTION_SCOPE
    assert first.env_config["initial_state_distribution"] == "all_good"
    assert first.gamma == pytest.approx(0.990)
    assert first.lambda_ == pytest.approx(0.95)
    assert first.vf_clip_param == pytest.approx(BEST_VF_CLIP_PARAM)
    assert first.vf_loss_coeff == pytest.approx(BEST_VF_LOSS_COEFF)
    assert first.entropy_coeff == pytest.approx(0.03)
    assert first.use_kl_loss is False
    assert first.rl_module_spec.model_config == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 64,
        "max_seq_len": 64,
    }
    assert first.num_env_runners == 0


@pytest.mark.parametrize("seed", [42, 43])
def test_previous_reward_recipe_accepts_both_study_seeds(tmp_path, seed):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=seed,
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    config = build_config(context)
    assert config.seed == seed


def test_previous_reward_long_run_budget():
    assert TOTAL_ENV_STEPS == 250_000_000
