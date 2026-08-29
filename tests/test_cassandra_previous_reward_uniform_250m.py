"""Focused tests for the 250M previous-reward uniform-start Cassandra recipe."""

from __future__ import annotations

import pytest

from experiments.cassandra_belief_factoring_2026_08.previous_reward_uniform_250m.d64_3layer_4head.experiment import (
    D_MODEL as D64,
    build_config as build_d64,
)
from experiments.cassandra_belief_factoring_2026_08.previous_reward_uniform_250m.d96_3layer_4head.experiment import (
    D_MODEL as D96,
    build_config as build_d96,
)
from experiments.cassandra_belief_factoring_2026_08.previous_reward_uniform_250m.shared import (
    BEST_VF_CLIP_PARAM,
    BEST_VF_LOSS_COEFF,
    CHECKPOINT_STEP_INTERVAL,
    CONTEXT_LEN,
    TOTAL_ENV_STEPS,
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


@pytest.mark.parametrize(
    ("builder", "d_model"),
    [
        (build_d64, D64),
        (build_d96, D96),
    ],
)
def test_recipe_sets_previous_reward_uniform_and_transformer(
    smoke_context,
    builder,
    d_model,
):
    first = builder(smoke_context)
    second = builder(smoke_context)

    assert first is not second
    assert first.seed == second.seed == 42
    assert first.env is CassandraPreviousRewardObservationEnv
    assert first.env_config["action_scope"] == "targeted"
    assert first.env_config["initial_state_distribution"] == "uniform"
    assert first.gamma == pytest.approx(0.990)
    assert first.lambda_ == pytest.approx(0.95)
    assert first.vf_clip_param == pytest.approx(BEST_VF_CLIP_PARAM)
    assert first.vf_loss_coeff == pytest.approx(BEST_VF_LOSS_COEFF)
    assert first.entropy_coeff == pytest.approx(0.03)
    assert first.use_kl_loss is False
    assert first.rl_module_spec.model_config == {
        "d_model": d_model,
        "n_layers": 3,
        "n_heads": 4,
        "context_len": CONTEXT_LEN,
        "max_seq_len": CONTEXT_LEN,
    }
    assert first.num_env_runners == (0 if smoke_context.smoke else 8)
    assert first.num_envs_per_env_runner == (1 if smoke_context.smoke else 2)


def test_long_run_budget_and_checkpoint_interval():
    assert TOTAL_ENV_STEPS == 250_000_000
    assert CHECKPOINT_STEP_INTERVAL == 50_000_000
