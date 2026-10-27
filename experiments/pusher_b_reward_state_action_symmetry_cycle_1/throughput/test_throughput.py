from __future__ import annotations

import importlib

import pytest

from experiments.pusher_b_reward_state_action_symmetry_cycle_1 import (
    shared as pusher_shared,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.shared import (
    MODEL_CONFIG,
    TRAIN_BATCH_SIZE,
    build_config as build_source_config,
)
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.throughput import (
    shared,
)
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
from harness.hardware import PROFILES


LAYOUTS = {
    16: (16, 130),
    32: (32, 65),
    52: (52, 40),
}


def _context(tmp_path, *, smoke: bool = False) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=smoke,
        hardware=PROFILES["cuda4090"],
    )


@pytest.mark.parametrize(
    ("max_env_runners", "expected"),
    LAYOUTS.items(),
)
def test_layouts_preserve_sampling_round_size(
    tmp_path,
    monkeypatch,
    max_env_runners,
    expected,
):
    monkeypatch.setattr(pusher_shared, "available_cpus", lambda: 144)
    context = _context(tmp_path)
    assert shared.sampling_layout(
        context,
        max_env_runners=max_env_runners,
    ) == expected
    assert expected[0] * expected[1] == 2_080


@pytest.mark.parametrize("max_env_runners", LAYOUTS)
def test_throughput_config_changes_only_sampling_parallelism(
    tmp_path,
    monkeypatch,
    max_env_runners,
):
    monkeypatch.setattr(pusher_shared, "available_cpus", lambda: 144)
    context = _context(tmp_path)
    source = build_source_config(
        context,
        preset=shared.PRESET,
        variant=shared.VARIANT,
        reward_state=shared.REWARD_STATE,
    )
    candidate = shared.build_config(
        context,
        max_env_runners=max_env_runners,
    )
    assert candidate is not shared.build_config(
        context,
        max_env_runners=max_env_runners,
    )
    assert candidate.env_config == source.env_config
    assert candidate.rl_module_spec.model_config == MODEL_CONFIG
    assert candidate.train_batch_size_per_learner == TRAIN_BATCH_SIZE
    assert candidate.minibatch_size == source.minibatch_size
    assert candidate.num_epochs == source.num_epochs
    assert candidate.lr == source.lr
    assert candidate.gamma == source.gamma
    assert candidate.lambda_ == source.lambda_
    assert candidate.clip_param == source.clip_param
    assert candidate.seed == source.seed == 42
    assert candidate.env_runner_cls is FreshEpisodeSingleAgentEnvRunner
    assert candidate.batch_mode == source.batch_mode == "complete_episodes"
    assert candidate.rollout_fragment_length == source.rollout_fragment_length
    assert candidate.num_gpus_per_env_runner == 0
    assert candidate.num_gpus_per_learner == source.num_gpus_per_learner == 1
    assert (
        candidate.num_env_runners,
        candidate.num_envs_per_env_runner,
    ) == LAYOUTS[max_env_runners]


def test_recipe_records_isolated_benchmark_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(pusher_shared, "available_cpus", lambda: 144)
    recipe = shared.resolved_recipe(
        _context(tmp_path),
        label="b10_reward_b_variant_2_cpu_52",
        max_env_runners=52,
    )
    assert recipe["study"].endswith("_throughput")
    assert recipe["source_recipe"].endswith(
        "b10_reward_b.variant_2.experiment"
    )
    assert recipe["total_env_steps"] == shared.BENCHMARK_ENV_STEPS
    assert recipe["sampling_layout"]["num_env_runners"] == 52
    assert recipe["sampling_layout"]["episodes_per_sampling_round"] == 2_080


@pytest.mark.parametrize("leaf", ["baseline_16", "cpu_32", "cpu_52"])
def test_throughput_leaves_are_importable(leaf):
    module = importlib.import_module(
        "experiments.pusher_b_reward_state_action_symmetry_cycle_1."
        f"throughput.{leaf}.experiment"
    )
    assert callable(module.run)
