"""Tests for deterministic first-episode desynchronization."""

from __future__ import annotations

import pytest
from ray.rllib.env.env_context import EnvContext

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    BEST_VF_CLIP_PARAM,
    BEST_VF_LOSS_COEFF,
    TOTAL_ENV_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted_previous_reward.experiment import (
    build_config as build_synchronized_config,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.targeted_previous_reward_desynced.experiment import (
    DESYNC_ENVS_PER_RUNNER_KEY,
    DESYNC_SEED_KEY,
    CassandraDesyncedPreviousRewardObservationEnv,
    build_config,
    initial_episode_horizon,
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


def test_desynced_recipe_changes_only_environment_adapter(smoke_context):
    synchronized = build_synchronized_config(smoke_context)
    desynchronized = build_config(smoke_context)

    assert desynchronized.env is CassandraDesyncedPreviousRewardObservationEnv
    assert desynchronized.env_config[DESYNC_SEED_KEY] == 42
    assert desynchronized.env_config[DESYNC_ENVS_PER_RUNNER_KEY] == 1
    assert desynchronized.env_config["action_scope"] == "targeted"
    assert desynchronized.env_config["initial_state_distribution"] == "all_good"
    assert desynchronized.gamma == synchronized.gamma == pytest.approx(0.990)
    assert desynchronized.lambda_ == synchronized.lambda_ == pytest.approx(0.95)
    assert (
        desynchronized.vf_clip_param
        == synchronized.vf_clip_param
        == pytest.approx(BEST_VF_CLIP_PARAM)
    )
    assert (
        desynchronized.vf_loss_coeff
        == synchronized.vf_loss_coeff
        == pytest.approx(BEST_VF_LOSS_COEFF)
    )
    assert (
        desynchronized.entropy_coeff
        == synchronized.entropy_coeff
        == pytest.approx(0.03)
    )
    assert (
        desynchronized.rl_module_spec.model_config
        == synchronized.rl_module_spec.model_config
        == {
            "d_model": 64,
            "n_layers": 4,
            "n_heads": 1,
            "context_len": 64,
            "max_seq_len": 64,
        }
    )
    assert TOTAL_ENV_STEPS == 250_000_000


def test_initial_horizons_evenly_cover_full_episode():
    horizons = [
        initial_episode_horizon(
            episode_length=1_000,
            seed=42,
            worker_index=worker_index,
            vector_index=vector_index,
            num_workers=16,
            envs_per_runner=4,
        )
        for worker_index in range(1, 17)
        for vector_index in range(4)
    ]

    assert len(set(horizons)) == 64
    phases = sorted(horizon - 1 for horizon in horizons)
    circular_gaps = [
        (phases[(index + 1) % len(phases)] - phase) % 1_000
        for index, phase in enumerate(phases)
    ]
    assert min(circular_gaps) >= 15
    assert max(circular_gaps) <= 16


def test_only_first_episode_uses_staggered_horizon():
    context = EnvContext(
        {
            "action_scope": "targeted",
            "episode_length": 10,
            "initial_state_distribution": "all_good",
            DESYNC_SEED_KEY: 0,
            DESYNC_ENVS_PER_RUNNER_KEY: 4,
        },
        worker_index=2,
        vector_index=0,
        num_workers=2,
    )
    env = CassandraDesyncedPreviousRewardObservationEnv(context)
    assert env.initial_episode_horizon == 6

    observation, _ = env.reset(seed=7)
    assert observation[-1] == pytest.approx(0.0)
    for step in range(1, 7):
        observation, reward, terminated, truncated, _ = env.step(0)
        assert observation[-1] == pytest.approx(reward)
        assert terminated is False
        assert truncated is (step == 6)

    with pytest.raises(RuntimeError, match="reset must be called"):
        env.step(0)

    env.reset(seed=8)
    for step in range(1, 11):
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is False
        assert truncated is (step == 10)
