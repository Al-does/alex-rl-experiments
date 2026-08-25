"""Focused tests for the Cassandra targeted PPO intervention grid."""

from __future__ import annotations

import numpy as np
import pytest

from envs.cassandra_machine import TargetedAction
from experiments.cassandra_belief_factoring_2026_08.environment import (
    TARGETED_OBSERVATION_DIM,
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.bptt_64.experiment import (
    build_config as build_bptt_64,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.lambda_098.experiment import (
    build_config as build_lambda_098,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.previous_reward.experiment import (
    build_config as build_previous_reward,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    CassandraPreviousRewardObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.vf_clip_100.experiment import (
    build_config as build_vf_clip_100,
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
    ("builder", "vf_clip", "gae_lambda", "context_len", "max_seq_len", "env"),
    [
        (
            build_vf_clip_100,
            100.0,
            0.95,
            256,
            256,
            CassandraActionObservationEnv,
        ),
        (
            build_lambda_098,
            10.0,
            0.98,
            256,
            256,
            CassandraActionObservationEnv,
        ),
        (
            build_bptt_64,
            10.0,
            0.95,
            64,
            64,
            CassandraActionObservationEnv,
        ),
        (
            build_previous_reward,
            10.0,
            0.95,
            256,
            256,
            CassandraPreviousRewardObservationEnv,
        ),
    ],
)
def test_intervention_changes_only_its_controlled_axis(
    smoke_context,
    builder,
    vf_clip,
    gae_lambda,
    context_len,
    max_seq_len,
    env,
):
    first = builder(smoke_context)
    second = builder(smoke_context)

    assert first is not second
    assert first.seed == second.seed == 42
    assert first.env is env
    assert first.env_config["action_scope"] == "targeted"
    assert first.env_config["initial_state_distribution"] == "all_good"
    assert first.gamma == pytest.approx(0.990)
    assert first.lambda_ == pytest.approx(gae_lambda)
    assert first.vf_clip_param == pytest.approx(vf_clip)
    assert first.entropy_coeff == pytest.approx(0.03)
    assert first.use_kl_loss is False
    assert first.kl_coeff == 0.0
    assert first.rl_module_spec.model_config == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": context_len,
        "max_seq_len": max_seq_len,
    }
    assert first.num_env_runners == 0


def test_previous_reward_is_visible_at_the_next_decision():
    environment = CassandraPreviousRewardObservationEnv(
        {
            "action_scope": "targeted",
            "episode_length": 3,
            "initial_state_distribution": "all_good",
            "diagnostics": True,
        }
    )
    try:
        observation, _ = environment.reset(seed=42)
        assert observation.shape == (TARGETED_OBSERVATION_DIM + 1,)
        assert observation[-1] == 0.0

        next_observation, reward, _, _, _ = environment.step(
            TargetedAction.REPLACE_COMPONENT_3
        )
        assert reward == -3.75
        assert next_observation[-1] == np.float32(reward)
        assert environment.observation_space.contains(next_observation)
    finally:
        environment.close()


def test_intervention_grid_rejects_mismatched_seed(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=7,
        smoke=True,
        hardware=PROFILES["cpu"],
    )

    with pytest.raises(ValueError, match="require seed 42"):
        build_lambda_098(context)
