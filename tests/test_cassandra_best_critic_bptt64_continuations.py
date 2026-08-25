"""Focused tests for targeted 250M continuations and entropy anneals."""

from __future__ import annotations

import pytest

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.checkpoint_recovery import (
    SOURCE_RUNS,
    source_run_id,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.continuation_shared import (
    ANNEAL_DURATION_ENV_STEPS,
    ANNEAL_FINAL_ENTROPY,
    ANNEAL_LIFETIME_ENV_STEPS,
    CONTINUE_LIFETIME_ENV_STEPS,
    ENTROPY_ANNEAL_SCHEDULE,
    PRIOR_LIFETIME_ENV_STEPS,
    build_anneal_config,
    build_continue_config,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    BEST_VF_CLIP_PARAM,
    BEST_VF_LOSS_COEFF,
    ENTROPY_COEFF,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted_continue_250m.experiment import (
    build_config as build_continue_leaf,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted_entropy_anneal_5m.experiment import (
    build_config as build_anneal_leaf,
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


def test_source_runs_cover_both_targeted_seeds():
    assert set(SOURCE_RUNS) == {42, 43}
    assert source_run_id(42) == "20260825T062526Z-0452c263"
    assert source_run_id(43) == "20260825T070342Z-fa83b069"


def test_lifetime_budgets():
    assert PRIOR_LIFETIME_ENV_STEPS == 250_000_000
    assert CONTINUE_LIFETIME_ENV_STEPS == 500_000_000
    assert ANNEAL_LIFETIME_ENV_STEPS == 255_000_000
    assert ANNEAL_DURATION_ENV_STEPS == 5_000_000


def test_continue_config_matches_best_critic_recipe(smoke_context):
    config = build_continue_leaf(smoke_context)
    assert config.vf_clip_param == pytest.approx(BEST_VF_CLIP_PARAM)
    assert config.vf_loss_coeff == pytest.approx(BEST_VF_LOSS_COEFF)
    assert config.entropy_coeff == pytest.approx(ENTROPY_COEFF)
    assert config.env_config["action_scope"] == "targeted"


def test_anneal_config_uses_five_million_step_schedule(smoke_context):
    config = build_anneal_leaf(smoke_context)
    assert config.entropy_coeff == ENTROPY_ANNEAL_SCHEDULE
    assert config.entropy_coeff[-1] == [ANNEAL_DURATION_ENV_STEPS, ANNEAL_FINAL_ENTROPY]
    assert config.vf_clip_param == pytest.approx(BEST_VF_CLIP_PARAM)


def test_builders_return_independent_configs(smoke_context):
    first = build_continue_config(smoke_context)
    second = build_continue_config(smoke_context)
    third = build_anneal_config(smoke_context)
    assert first is not second
    assert first.entropy_coeff == pytest.approx(ENTROPY_COEFF)
    assert third.entropy_coeff == ENTROPY_ANNEAL_SCHEDULE
