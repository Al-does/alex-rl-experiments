"""Focused tests for the 250M best-critic BPTT-64 Cassandra recipe."""

from __future__ import annotations

import pytest

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.global_alias.experiment import (
    ACTION_SCOPE as GLOBAL_SCOPE,
    build_config as build_global,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    BEST_VF_CLIP_PARAM,
    BEST_VF_LOSS_COEFF,
    CHECKPOINT_STEP_INTERVAL,
    TOTAL_ENV_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted.experiment import (
    ACTION_SCOPE as TARGETED_SCOPE,
    build_config as build_targeted,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
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
    ("builder", "action_scope"),
    [
        (build_global, GLOBAL_SCOPE),
        (build_targeted, TARGETED_SCOPE),
    ],
)
def test_recipe_sets_best_critic_bptt64_and_scope(
    smoke_context,
    builder,
    action_scope,
):
    first = builder(smoke_context)
    second = builder(smoke_context)

    assert first is not second
    assert first.seed == second.seed == 42
    assert first.env is CassandraActionObservationEnv
    assert first.env_config["action_scope"] == action_scope
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


def test_long_run_budget_and_checkpoint_interval():
    assert TOTAL_ENV_STEPS == 250_000_000
    assert CHECKPOINT_STEP_INTERVAL == 50_000_000
