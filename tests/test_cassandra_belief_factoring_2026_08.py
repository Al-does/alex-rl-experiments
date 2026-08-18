"""Focused tests for the Cassandra transformer belief-factoring study."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from envs.cassandra_machine import (
    N_COMPONENTS,
    N_CONDITIONS,
    N_STATES,
    Action,
)
from experiments.cassandra_belief_factoring_2026_08.analysis import (
    factor_subspace_geometry,
    variance_geometry,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    OBSERVATION_DIM,
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.probe import (
    belief_targets,
    collect_probe_data,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    SMOKE_BATCH_SIZE,
    build_config,
    log_spaced_records,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models import TransformerModel


def test_history_observation_exposes_symbol_and_preceding_action():
    environment = CassandraActionObservationEnv(
        {"episode_length": 3, "diagnostics": True}
    )
    try:
        observation, info = environment.reset(seed=42)
        assert observation.shape == (OBSERVATION_DIM,)
        assert observation[:16].sum() == 1.0
        assert observation[16:].sum() == 0.0
        assert "belief_current" in info

        next_observation, _, _, _, next_info = environment.step(
            Action.INSPECT
        )
        assert next_observation[:16].sum() == 1.0
        assert next_observation[16:].sum() == 1.0
        assert next_observation[16 + int(Action.INSPECT)] == 1.0
        assert next_info["action"] == int(Action.INSPECT)
    finally:
        environment.close()


def test_belief_targets_separate_aggregate_and_identity_information():
    joint = np.zeros((1, N_STATES), dtype=np.float64)
    joint[0, -1] = 1.0
    marginals = np.zeros(
        (1, N_COMPONENTS, N_CONDITIONS),
        dtype=np.float64,
    )
    marginals[:, :, -1] = 1.0

    targets = belief_targets(joint, marginals)

    assert targets["component_contrast"].shape == (1, 12)
    assert targets["aggregate_contrast"].shape == (1, 3)
    np.testing.assert_allclose(targets["identity_deviation"], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        targets["labeled_expected_condition"],
        3.0,
    )
    np.testing.assert_allclose(
        targets["broken_count_distribution"],
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(targets["total_correlation"], 0.0, atol=1e-12)


def test_pca_and_component_subspaces_recover_known_factored_geometry():
    rng = np.random.default_rng(42)
    activations = rng.normal(size=(2_000, 12))
    component_targets = activations.copy()

    pca = variance_geometry(activations)
    subspaces = factor_subspace_geometry(activations, component_targets)

    assert pca["rank"] == 12
    assert 10 <= pca["cev95_dimension"] <= 12
    assert subspaces["component_ranks"] == [3, 3, 3, 3]
    assert subspaces["union_rank"] == 12
    assert subspaces["mean_pairwise_overlap"] < 1e-6


def test_probe_histories_and_targets_are_checkpoint_independent():
    config = {"episode_length": 12, "diagnostics": True}

    def make_environment():
        return CassandraActionObservationEnv(config)

    environment = make_environment()
    try:
        spaces = (environment.observation_space, environment.action_space)
    finally:
        environment.close()
    model_config = {
        "context_len": 4,
        "d_model": 24,
        "n_layers": 1,
        "n_heads": 3,
        "max_seq_len": 4,
    }
    torch.manual_seed(1)
    first_module = TransformerModel(
        observation_space=spaces[0],
        action_space=spaces[1],
        model_config=model_config,
    )
    torch.manual_seed(2)
    second_module = TransformerModel(
        observation_space=spaces[0],
        action_space=spaces[1],
        model_config=model_config,
    )

    first = collect_probe_data(
        first_module,
        make_environment,
        n_steps=32,
        seed=42,
        n_envs=2,
        warmup=1,
    )
    second = collect_probe_data(
        second_module,
        make_environment,
        n_steps=32,
        seed=42,
        n_envs=2,
        warmup=1,
    )

    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.observations, second.observations)
    for name in first.targets:
        np.testing.assert_allclose(first.targets[name], second.targets[name])
    assert first.marginal_consistency_max_abs < 1e-12
    assert second.marginal_consistency_max_abs < 1e-12
    assert not np.allclose(first.activations, second.activations)


def test_smoke_recipe_builds_fresh_canonical_configs(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )

    first = build_config(context)
    second = build_config(context)

    assert first is not second
    assert first.gamma == 0.999
    assert first.seed == 42
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size == 256
    assert first.num_env_runners == 0
    environment = first.env(first.env_config)
    try:
        assert environment.observation_space.shape == (OBSERVATION_DIM,)
        assert environment.action_space.n == 4
    finally:
        environment.close()


def test_log_schedule_keeps_powers_of_two_and_final():
    records = [
        {
            "checkpoint": SimpleNamespace(path=f"/tmp/checkpoint_{iteration}"),
            "checkpoint_name": f"checkpoint_{iteration}",
            "training_iteration": iteration,
            "agent_steps": iteration * 2_048,
        }
        for iteration in range(1, 7)
    ]

    selected = log_spaced_records(records)

    assert [record["training_iteration"] for record in selected] == [
        1,
        2,
        4,
        6,
    ]
