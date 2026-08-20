"""Focused tests for the Cassandra transformer belief-factoring study."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ray.rllib.utils.postprocessing.zero_padding import create_mask_and_seq_lens
import torch

from envs.cassandra_machine import (
    N_COMPONENTS,
    N_CONDITIONS,
    N_STATES,
    Action,
    GlobalAliasAction,
    TargetedAction,
)
from experiments.cassandra_belief_factoring_2026_08.analysis import (
    factor_subspace_geometry,
    variance_geometry,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    OBSERVATION_DIM,
    TARGETED_OBSERVATION_DIM,
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.probe import (
    belief_targets,
    collect_probe_data,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    MINIBATCH_SIZE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
    TRAIN_ENVS_PER_ENV_RUNNER,
    _last_reported_return,
    _save_log_spaced_checkpoint,
    checkpoint_records,
    log_spaced_records,
    training_curve,
)
from experiments.cassandra_belief_factoring_2026_08.small_final_checkpoint_probes.experiment import (
    AGENT_STEPS as SMALL_FINAL_PROBE_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.small_final_checkpoint_probes.experiment import (
    checkpoint_paths as small_final_checkpoint_paths,
)
from experiments.cassandra_belief_factoring_2026_08.global_alias_ppo.experiment import (
    build_config as build_global_alias_config,
)
from experiments.cassandra_belief_factoring_2026_08.global_alias_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m.experiment import (
    build_config as build_global_alias_small_config,
)
from experiments.cassandra_belief_factoring_2026_08.global_alias_ppo_small_entropy_anneal_continue_10m.experiment import (
    build_config as build_global_alias_small_continuation_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo.experiment import (
    build_config as build_targeted_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_03_gamma_0_990_all_good_10m.experiment import (
    build_config as build_entropy_003_standard_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m.experiment import (
    MODEL_CONFIG as SMALL_FOUR_LAYER_MODEL_CONFIG,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m.experiment import (
    build_config as build_entropy_003_small_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_05_gamma_0_990_all_good_10m.experiment import (
    build_config as build_entropy_005_standard_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_08_all_good.experiment import (
    ENTROPY_COEFF as MODERATE_ENTROPY_COEFF,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_08_all_good.experiment import (
    build_config as build_moderate_entropy_targeted_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_8_all_good.experiment import (
    ENTROPY_COEFF as HIGH_ENTROPY_COEFF,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_0_8_all_good.experiment import (
    build_config as build_high_entropy_targeted_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_anneal_all_good.experiment import (
    ENTROPY_COEFF_SCHEDULE,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_anneal_all_good.experiment import (
    build_config as build_annealed_entropy_targeted_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_anneal_continue_5m.experiment import (
    ADDITIONAL_ENV_STEPS,
    SOURCE_STEPS,
    TARGET_ENV_STEPS,
    _sampled_steps,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_entropy_anneal_continue_5m.experiment import (
    build_config as build_entropy_continuation_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_entropy_anneal_continue_10m.experiment import (
    ADDITIONAL_ENV_STEPS as SMALL_CONTINUATION_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_entropy_anneal_continue_10m.experiment import (
    ANNEAL_END_STEPS as SMALL_ANNEAL_END_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_entropy_anneal_continue_10m.experiment import (
    ENTROPY_COEFF_SCHEDULE as SMALL_CONTINUATION_ENTROPY_SCHEDULE,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_entropy_anneal_continue_10m.experiment import (
    SOURCE_STEPS as SMALL_SOURCE_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_entropy_anneal_continue_10m.experiment import (
    build_config as build_targeted_small_continuation_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models import TransformerModel


def _padded_learner_sequences(config, *, episode_phase: int) -> int:
    """Model RLlib's bootstrap-timestep and stateful zero-padding semantics."""

    max_seq_len = MODEL_CONFIG["max_seq_len"]
    episode_length = int(config.env_config["episode_length"])
    total = 0
    for worker_index in range(1, config.num_env_runners + 1):
        remaining = config.get_rollout_fragment_length(worker_index)
        position = episode_phase
        rows_per_environment = 0
        while remaining:
            chunk_length = min(remaining, episode_length - position)
            # PPO adds one bootstrap timestep to every returned episode chunk
            # before RLlib splits and zero-pads it to max_seq_len.
            _, seq_lens = create_mask_and_seq_lens(
                chunk_length + 1,
                max_seq_len,
            )
            rows_per_environment += len(seq_lens)
            remaining -= chunk_length
            position = (position + chunk_length) % episode_length
        total += rows_per_environment * config.num_envs_per_env_runner
    return total


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


def test_targeted_history_observation_exposes_ten_action_features():
    environment = CassandraActionObservationEnv(
        {
            "action_scope": "targeted",
            "episode_length": 3,
            "diagnostics": True,
        }
    )
    try:
        observation, _ = environment.reset(seed=42)
        assert observation.shape == (TARGETED_OBSERVATION_DIM,)
        assert observation[16:].sum() == 0.0

        next_observation, reward, _, _, info = environment.step(
            TargetedAction.REPLACE_COMPONENT_3
        )
        assert reward == -3.75
        assert next_observation[16:].sum() == 1.0
        assert (
            next_observation[
                16 + int(TargetedAction.REPLACE_COMPONENT_3)
            ]
            == 1.0
        )
        assert info["action_name"] == "replace_component_3"
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
    targeted = belief_targets(
        joint,
        marginals,
        action_scope="targeted",
    )
    assert targeted["expected_action_reward"].shape == (
        1,
        len(TargetedAction),
    )
    aliases = belief_targets(
        joint,
        marginals,
        action_scope="global_aliases",
    )
    assert aliases["expected_action_reward"].shape == (
        1,
        len(GlobalAliasAction),
    )
    np.testing.assert_array_equal(
        aliases["expected_action_reward"][:, 2:6],
        np.repeat(
            aliases["expected_action_reward"][:, 2:3],
            N_COMPONENTS,
            axis=1,
        ),
    )
    np.testing.assert_array_equal(
        aliases["expected_action_reward"][:, 6:10],
        np.repeat(
            aliases["expected_action_reward"][:, 6:7],
            N_COMPONENTS,
            axis=1,
        ),
    )


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


@pytest.mark.parametrize(
    "action_scope",
    ["global_aliases", "targeted"],
)
def test_probe_histories_and_targets_are_checkpoint_independent(action_scope):
    config = {
        "action_scope": action_scope,
        "episode_length": 12,
        "diagnostics": True,
    }

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
        action_scope=action_scope,
    )
    second = collect_probe_data(
        second_module,
        make_environment,
        n_steps=32,
        seed=42,
        n_envs=2,
        warmup=1,
        action_scope=action_scope,
    )

    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.observations, second.observations)
    for name in first.targets:
        np.testing.assert_allclose(first.targets[name], second.targets[name])
    assert first.marginal_consistency_max_abs < 1e-12
    assert second.marginal_consistency_max_abs < 1e-12
    assert not np.allclose(first.activations, second.activations)


def test_smoke_recipe_builds_fresh_global_alias_configs(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )

    first = build_global_alias_config(context)
    second = build_global_alias_config(context)

    assert first is not second
    assert first.gamma == 0.999
    assert first.seed == 42
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size == 256
    assert first.num_env_runners == 0
    assert first.env_config["action_scope"] == "global_aliases"
    assert first.env_config["initial_state_distribution"] == "uniform"
    assert MODEL_CONFIG["context_len"] == 256
    assert MODEL_CONFIG["max_seq_len"] == 256
    environment = first.env(first.env_config)
    try:
        assert environment.observation_space.shape == (
            TARGETED_OBSERVATION_DIM,
        )
        assert environment.action_space.n == len(GlobalAliasAction)
    finally:
        environment.close()


def test_smoke_recipe_builds_targeted_config(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )

    config = build_targeted_config(context)
    environment = config.env(config.env_config)
    try:
        assert config.env_config["action_scope"] == "targeted"
        assert config.env_config["initial_state_distribution"] == "uniform"
        assert config.rl_module_spec.model_config["context_len"] == 256
        assert config.rl_module_spec.model_config["max_seq_len"] == 256
        assert environment.observation_space.shape == (
            TARGETED_OBSERVATION_DIM,
        )
        assert environment.action_space.n == len(TargetedAction)
    finally:
        environment.close()


def test_high_entropy_targeted_recipe_uses_all_good_starts(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    config = build_high_entropy_targeted_config(context)

    assert HIGH_ENTROPY_COEFF == 0.8
    assert config.entropy_coeff == 0.8
    assert config.env_config["action_scope"] == "targeted"
    assert config.env_config["initial_state_distribution"] == "all_good"
    assert config.train_batch_size_per_learner == TRAIN_BATCH_SIZE
    assert config.minibatch_size == MINIBATCH_SIZE


def test_moderate_entropy_targeted_recipe_uses_all_good_starts(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    config = build_moderate_entropy_targeted_config(context)

    assert MODERATE_ENTROPY_COEFF == 0.08
    assert config.entropy_coeff == 0.08
    assert config.env_config["action_scope"] == "targeted"
    assert config.env_config["initial_state_distribution"] == "all_good"
    assert config.train_batch_size_per_learner == TRAIN_BATCH_SIZE
    assert config.minibatch_size == MINIBATCH_SIZE


def test_annealed_entropy_targeted_recipe_uses_expected_schedule(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    config = build_annealed_entropy_targeted_config(context)

    assert ENTROPY_COEFF_SCHEDULE == [
        [0, 0.08],
        [2_500_000, 0.08],
        [5_000_000, 0.01],
    ]
    assert config.entropy_coeff == ENTROPY_COEFF_SCHEDULE
    assert config.entropy_coeff_schedule is None
    assert config.env_config["action_scope"] == "targeted"
    assert config.env_config["initial_state_distribution"] == "all_good"


def test_entropy_continuation_targets_five_million_additional_steps(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    config = build_entropy_continuation_config(context)

    assert SOURCE_STEPS == 5_013_504
    assert ADDITIONAL_ENV_STEPS == 5_000_000
    assert TARGET_ENV_STEPS == 10_013_504
    assert config.entropy_coeff[-1] == [5_000_000, 0.01]
    assert config.env_config["initial_state_distribution"] == "all_good"
    assert _sampled_steps(
        {"env_runners": {"num_env_steps_sampled_lifetime": 10_027_008}}
    ) == 10_027_008


def test_entropy_003_10m_recipes_match_except_for_transformer_shape(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    standard = build_entropy_003_standard_config(context)
    small = build_entropy_003_small_config(context)

    for config in (standard, small):
        assert config.entropy_coeff == 0.03
        assert config.gamma == 0.990
        assert config.use_kl_loss is False
        assert config.kl_coeff == 0.0
        assert config.env_config["initial_state_distribution"] == "all_good"
        assert config.train_batch_size_per_learner == TRAIN_BATCH_SIZE
        assert config.minibatch_size == MINIBATCH_SIZE
    assert standard.rl_module_spec.model_config == MODEL_CONFIG
    assert small.rl_module_spec.model_config == SMALL_FOUR_LAYER_MODEL_CONFIG
    assert SMALL_FOUR_LAYER_MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 256,
        "max_seq_len": 256,
    }


def test_entropy_005_10m_recipe_keeps_standard_transformer(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    config = build_entropy_005_standard_config(context)

    assert config.entropy_coeff == 0.05
    assert config.gamma == 0.990
    assert config.use_kl_loss is False
    assert config.kl_coeff == 0.0
    assert config.env_config["initial_state_distribution"] == "all_good"
    assert config.rl_module_spec.model_config == MODEL_CONFIG


def test_small_global_alias_recipe_matches_small_targeted_architecture(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    global_alias = build_global_alias_small_config(context)
    targeted = build_entropy_003_small_config(context)

    assert global_alias.env_config["action_scope"] == "global_aliases"
    assert targeted.env_config["action_scope"] == "targeted"
    for config in (global_alias, targeted):
        assert config.entropy_coeff == 0.03
        assert config.gamma == 0.990
        assert config.use_kl_loss is False
        assert config.env_config["initial_state_distribution"] == "all_good"
        assert config.rl_module_spec.model_config == SMALL_FOUR_LAYER_MODEL_CONFIG


def test_small_final_probe_paths_resolve_under_study_root(tmp_path):
    study_root = tmp_path / "cassandra_belief_factoring_2026_08"
    context = RunContext(
        experiment_dir=study_root / "small_final_checkpoint_probes",
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )

    paths = small_final_checkpoint_paths(context)

    assert SMALL_FINAL_PROBE_STEPS == 10_027_008
    assert set(paths) == {"targeted", "global_alias"}
    assert all(path.is_relative_to(study_root) for path in paths.values())
    assert all(path.name == "checkpoint_000000" for path in paths.values())


def test_small_continuations_share_lifetime_entropy_schedule(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )

    targeted = build_targeted_small_continuation_config(context)
    global_alias = build_global_alias_small_continuation_config(context)

    assert SMALL_SOURCE_STEPS == 10_027_008
    assert SMALL_CONTINUATION_STEPS == 10_000_000
    assert SMALL_ANNEAL_END_STEPS == 12_527_008
    assert SMALL_CONTINUATION_ENTROPY_SCHEDULE == [
        [0, 0.03],
        [10_027_008, 0.03],
        [12_527_008, 0.008],
    ]
    for config in (targeted, global_alias):
        assert config.entropy_coeff == SMALL_CONTINUATION_ENTROPY_SCHEDULE
        assert config.gamma == 0.990
        assert config.use_kl_loss is False
        assert config.rl_module_spec.model_config == SMALL_FOUR_LAYER_MODEL_CONFIG
    assert targeted.env_config["action_scope"] == "targeted"
    assert global_alias.env_config["action_scope"] == "global_aliases"


def test_full_recipe_limits_stateful_connector_fanout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("harness.hardware.available_cpus", lambda: 64.0)
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090"],
    )

    config = build_global_alias_config(context)

    assert TRAIN_BATCH_SIZE == 32_768
    assert MINIBATCH_SIZE == 8_192
    assert TRAIN_ENVS_PER_ENV_RUNNER == 4
    assert config.train_batch_size_per_learner == 32_768
    assert config.minibatch_size == 8_192
    assert config.num_env_runners == 16
    assert config.num_envs_per_env_runner == 4
    sequence_counts = [
        _padded_learner_sequences(config, episode_phase=phase)
        for phase in range(config.env_config["episode_length"])
    ]
    assert min(sequence_counts) == 192
    assert max(sequence_counts) == 256


def test_previous_vectorization_reproduces_episode_boundary_sequence_spike(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("harness.hardware.available_cpus", lambda: 64.0)
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090"],
    )
    config = build_global_alias_config(context)
    config.train_batch_size_per_learner = 1_024
    config.num_envs_per_env_runner = 24

    sequence_counts = [
        _padded_learner_sequences(config, episode_phase=phase)
        for phase in range(config.env_config["episode_length"])
    ]

    assert min(sequence_counts) == 384
    assert max(sequence_counts) == 768
    assert max(sequence_counts) >= 720


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


def test_training_curve_preserves_null_windows_and_finds_last_return():
    rows = [
        {
            "training_iteration": 1,
            "env_runners/num_env_steps_sampled_lifetime": 2_048,
            "env_runners/num_episodes": 2,
            "env_runners/episode_return_mean": 3.5,
            "env_runners/episode_len_mean": 1_000,
        },
        {
            "training_iteration": 2,
            "env_runners/num_env_steps_sampled_lifetime": 4_096,
            "env_runners/num_episodes": 0,
            "env_runners/episode_return_mean": np.nan,
            "env_runners/episode_len_mean": np.nan,
        },
    ]
    dataframe = SimpleNamespace(
        iterrows=lambda: [
            (index, SimpleNamespace(to_dict=lambda row=row: row))
            for index, row in enumerate(rows)
        ]
    )

    curve = training_curve(SimpleNamespace(metrics_dataframe=dataframe))

    assert len(curve) == 2
    assert curve[-1]["episode_return_mean"] is None
    assert _last_reported_return(curve) == curve[0]


def test_log_spaced_callback_saves_only_power_of_two_iterations(tmp_path):
    saved = []

    class Algorithm:
        def save_to_path(self, path):
            Path(path).mkdir(parents=True)
            saved.append(path)
            return path

    for iteration in (1, 2, 3, 4):
        _save_log_spaced_checkpoint(
            algorithm=Algorithm(),
            result={
                "training_iteration": iteration,
                "env_runners": {
                    "num_env_steps_sampled_lifetime": iteration * 2_048
                },
            },
            checkpoint_root=str(tmp_path),
        )

    index = json.loads((tmp_path / "index.json").read_text())
    assert [row["training_iteration"] for row in index["checkpoints"]] == [
        1,
        2,
        4,
    ]
    assert len(saved) == 3


def test_checkpoint_records_combine_log_spaced_and_final(tmp_path):
    root = tmp_path / "checkpoints"
    first = root / "iteration_000001_steps_000002048"
    first.mkdir(parents=True)
    (root / "index.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "path": str(first),
                        "checkpoint_name": first.name,
                        "training_iteration": 1,
                        "agent_steps": 2_048,
                    }
                ]
            }
        )
    )
    final = SimpleNamespace(path=str(tmp_path / "checkpoint_final"))
    result = SimpleNamespace(
        best_checkpoints=[],
        checkpoint=final,
        metrics={
            "training_iteration": 3,
            "env_runners/num_env_steps_sampled_lifetime": 6_144,
        },
    )

    records = checkpoint_records(result, checkpoint_root=root)

    assert [row["training_iteration"] for row in records] == [1, 3]
    assert records[-1]["checkpoint_path"] == Path(final.path)
