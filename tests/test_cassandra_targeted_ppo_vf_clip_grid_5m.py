"""Focused tests for the Cassandra vf-clip / vf-loss-coeff grid."""

from __future__ import annotations

import pytest

from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_vf_clip_grid_5m.vf100_coeff0002.experiment import (
    build_config as build_vf100_coeff0002,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_vf_clip_grid_5m.vf100_coeff001.experiment import (
    build_config as build_vf100_coeff001,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_vf_clip_grid_5m.vf100_coeff005.experiment import (
    build_config as build_vf100_coeff005,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_vf_clip_grid_5m.vf400_coeff00025.experiment import (
    build_config as build_vf400_coeff00025,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_vf_clip_grid_5m.vf400_coeff00125.experiment import (
    build_config as build_vf400_coeff00125,
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
    ("builder", "vf_clip", "vf_loss_coeff"),
    [
        (build_vf100_coeff005, 100.0, 0.05),
        (build_vf100_coeff001, 100.0, 0.01),
        (build_vf100_coeff0002, 100.0, 0.002),
        (build_vf400_coeff00125, 400.0, 0.0125),
        (build_vf400_coeff00025, 400.0, 0.0025),
    ],
)
def test_grid_cell_sets_vf_clip_and_coeff_only(
    smoke_context,
    builder,
    vf_clip,
    vf_loss_coeff,
):
    first = builder(smoke_context)
    second = builder(smoke_context)

    assert first is not second
    assert first.seed == second.seed == 42
    assert first.env is CassandraActionObservationEnv
    assert first.env_config["action_scope"] == "targeted"
    assert first.env_config["initial_state_distribution"] == "all_good"
    assert first.gamma == pytest.approx(0.990)
    assert first.lambda_ == pytest.approx(0.95)
    assert first.vf_clip_param == pytest.approx(vf_clip)
    assert first.vf_loss_coeff == pytest.approx(vf_loss_coeff)
    assert first.entropy_coeff == pytest.approx(0.03)
    assert first.use_kl_loss is False
    assert first.rl_module_spec.model_config == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 256,
        "max_seq_len": 256,
    }
    assert first.num_env_runners == 0
