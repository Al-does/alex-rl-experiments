"""Focused tests for targeted 250M continuations and entropy anneals."""

from __future__ import annotations

from pathlib import Path

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
    apply_entropy_coeff_schedule,
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
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted_entropy_anneal_250m.experiment import (
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
    assert ANNEAL_LIFETIME_ENV_STEPS == 500_000_000
    assert ANNEAL_DURATION_ENV_STEPS == 250_000_000


def test_entropy_anneal_schedule_uses_lifetime_knots():
    assert ENTROPY_ANNEAL_SCHEDULE == [
        [0, ENTROPY_COEFF],
        [PRIOR_LIFETIME_ENV_STEPS, ENTROPY_COEFF],
        [ANNEAL_LIFETIME_ENV_STEPS, ANNEAL_FINAL_ENTROPY],
    ]


def test_continue_config_matches_best_critic_recipe(smoke_context):
    config = build_continue_leaf(smoke_context)
    assert config.vf_clip_param == pytest.approx(BEST_VF_CLIP_PARAM)
    assert config.vf_loss_coeff == pytest.approx(BEST_VF_LOSS_COEFF)
    assert config.entropy_coeff == pytest.approx(ENTROPY_COEFF)
    assert config.env_config["action_scope"] == "targeted"


def test_anneal_config_uses_two_hundred_fifty_million_step_schedule(smoke_context):
    config = build_anneal_leaf(smoke_context)
    assert config.entropy_coeff == ENTROPY_ANNEAL_SCHEDULE
    assert config.entropy_coeff[-1] == [
        ANNEAL_LIFETIME_ENV_STEPS,
        ANNEAL_FINAL_ENTROPY,
    ]
    assert config.vf_clip_param == pytest.approx(BEST_VF_CLIP_PARAM)


def test_builders_return_independent_configs(smoke_context):
    first = build_continue_config(smoke_context)
    second = build_continue_config(smoke_context)
    third = build_anneal_config(smoke_context)
    assert first is not second
    assert first.entropy_coeff == pytest.approx(ENTROPY_COEFF)
    assert third.entropy_coeff == ENTROPY_ANNEAL_SCHEDULE


@pytest.mark.slow
def test_apply_entropy_coeff_schedule_replaces_restored_fixed_scheduler():
    checkpoint = Path(
        "experiments/cassandra_belief_factoring_2026_08/"
        "best_critic_bptt64_250m/targeted_continue_250m/.smoke/"
        "20260825T173839Z-19923bf1/artifacts/tune/"
        "PPO_CassandraActionObservationEnv_cfeb4_00000_0_2026-08-25_17-38-41/"
        "checkpoint_000000"
    ).resolve()
    if not checkpoint.exists():
        pytest.skip("smoke checkpoint unavailable")

    from ray.rllib.algorithms.algorithm import Algorithm

    algorithm = Algorithm.from_checkpoint(str(checkpoint))
    try:
        learner = algorithm.learner_group._learner
        module_id = next(iter(learner.module.keys()))
        restored = learner.entropy_coeff_schedulers_per_module[module_id]
        assert restored.use_schedule is False

        apply_entropy_coeff_schedule(
            algorithm,
            ENTROPY_ANNEAL_SCHEDULE,
            lifetime_steps=PRIOR_LIFETIME_ENV_STEPS,
        )
        scheduled = learner.entropy_coeff_schedulers_per_module[module_id]
        assert scheduled.use_schedule is True
        assert scheduled.update(timestep=PRIOR_LIFETIME_ENV_STEPS) == pytest.approx(
            ENTROPY_COEFF
        )
        midpoint = PRIOR_LIFETIME_ENV_STEPS + ANNEAL_DURATION_ENV_STEPS // 2
        mid_coeff = scheduled.update(timestep=midpoint)
        assert ANNEAL_FINAL_ENTROPY < mid_coeff < ENTROPY_COEFF
        assert scheduled.update(timestep=ANNEAL_LIFETIME_ENV_STEPS) == pytest.approx(
            ANNEAL_FINAL_ENTROPY
        )
    finally:
        algorithm.stop()
