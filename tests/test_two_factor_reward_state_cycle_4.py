"""Recipe tests for baseline-free two-factor REINFORCE cycle 4."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

from envs.hmm import HMMEnv
from experiments.two_factor_reward_state_PPO_cycle_2.task import CONDITIONS
from experiments.two_factor_reward_state_REINFORCE_cycle_3.shared import (
    build_config as build_cycle_3_config,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.model import (
    TwoFactorRewardReinforceCycle4,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    LEARNING_RATE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    TOTAL_ENV_STEPS,
)
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_cycle_4_preserves_task_and_builds_simple_reinforce_recipe(
    tmp_path,
    condition,
):
    module = importlib.import_module(
        "experiments.two_factor_reward_state_REINFORCE_cycle_4."
        f"{condition}.experiment"
    )
    first = module.build_config(_context(tmp_path))
    second = module.build_config(_context(tmp_path))
    cycle_3 = build_cycle_3_config(_context(tmp_path), condition)
    spec = first.get_rl_module_spec()

    assert first is not second
    assert first.env_config == cycle_3.env_config
    assert first.env_config["task"]["kwargs"]["condition"] == condition
    assert first.seed == 42
    assert first.num_env_runners == 0
    assert first.lr == LEARNING_RATE == 4.2e-4
    assert first.gamma == 0.99
    assert first.lambda_ == 1.0
    assert first.use_critic is False
    assert first.use_gae is False
    assert first.use_kl_loss is False
    assert first.vf_loss_coeff == 0.0
    assert first.entropy_coeff == 0.0
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size is None
    assert first.num_epochs == 1
    assert first.batch_mode == "complete_episodes"
    assert spec.module_class is TwoFactorRewardReinforceCycle4
    first.validate()

    environment = HMMEnv(first.env_config)
    try:
        assert environment.observation_space.shape == (18,)
        assert environment.action_space.n == 9
    finally:
        environment.close()


def test_cycle_4_preserves_architecture_and_budget():
    assert TOTAL_ENV_STEPS == 8_000_000
    assert MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 10,
        "max_seq_len": 32,
    }


def test_cycle_4_resolve_step_target_uses_continuation_spec(tmp_path):
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
        CONTINUATION_SPEC_FILENAME,
        _resolve_step_target,
    )

    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )
    context.artifacts_dir.mkdir(parents=True)
    (context.artifacts_dir / CONTINUATION_SPEC_FILENAME).write_text(
        '{"target_agent_steps": 16000000}'
    )
    assert _resolve_step_target(context) == 16_000_000


def test_cycle_4_metric_reads_nested_lifetime_steps():
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
        _metric,
        _reached_env_step_target,
    )

    nested = {
        "env_runners": {"num_env_steps_sampled_lifetime": 16_000_000.0},
        "training_iteration": 42,
    }
    assert _metric(nested, "env_runners/num_env_steps_sampled_lifetime") == 16_000_000.0
    assert _reached_env_step_target(16_000_000)(nested) is True
    assert _reached_env_step_target(16_000_001)(nested) is False


def test_cycle_4_budget_spec_overrides_step_target(tmp_path):
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
        BUDGET_SPEC_FILENAME,
        _resolve_step_target,
        write_budget_spec,
    )

    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=43,
        smoke=False,
        hardware=PROFILES["cpu"],
    )
    write_budget_spec(context, 10_000_000)
    assert (context.artifacts_dir / BUDGET_SPEC_FILENAME).is_file()
    assert _resolve_step_target(context) == 10_000_000


def test_cycle_4_latest_algorithm_checkpoint_from_step_checkpoints(tmp_path):
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.continue_prior import (
        _latest_algorithm_checkpoint,
    )

    artifacts_root = tmp_path / "artifacts" / "prior"
    for steps in (8_000_000, 16_000_000):
        destination = artifacts_root / "step_checkpoints" / f"steps_{steps:09d}"
        destination.mkdir(parents=True)
        (destination / "rllib_checkpoint.json").write_text("{}")
    resolved = _latest_algorithm_checkpoint(artifacts_root)
    assert resolved.name == "steps_016000000"


def test_cycle_4_prior_agent_steps_from_condition_summary(tmp_path):
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.continue_prior import (
        _prior_agent_steps,
    )

    results_dir = tmp_path / "results" / "prior"
    results_dir.mkdir(parents=True)
    (results_dir / "condition_summary.json").write_text(
        json.dumps(
            {
                "checkpoint_reports": [
                    {"agent_steps": 10_000_000},
                    {"agent_steps": 16_059_680},
                ]
            }
        )
    )
    assert _prior_agent_steps(results_dir) == 16_059_680


def test_cycle_4_resume_checkpoint_path_resolution():
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.continue_prior import (
        _resume_checkpoint,
    )

    checkpoint = (
        "/root/work/alex-rl-experiments/experiments/"
        "two_factor_reward_state_REINFORCE_cycle_4/reward_both/"
        "artifacts/20260901T211811Z-3016ddd1/"
        "tune/PPO_HMMEnv_a529e_00000_0_2026-09-01_21-18-15/checkpoint_000000"
    )
    resolved = _resume_checkpoint(
        experiment_dir=Path(
            "/root/work/alex-rl-experiments/experiments/"
            "two_factor_reward_state_REINFORCE_cycle_4/reward_both"
        ),
        prior_run_id="20260901T211811Z-3016ddd1",
        checkpoint_remote=checkpoint,
    )
    assert resolved.name == "checkpoint_000000"
    assert resolved.parent.name.startswith("PPO_HMMEnv_")


def test_cycle_4_value_api_is_an_inert_device_native_zero_baseline():
    embeddings = torch.randn(3, 5, 64)
    values = TwoFactorRewardReinforceCycle4.compute_values(
        object(),
        {},
        embeddings=embeddings,
    )

    assert values.shape == (3, 5)
    assert values.device == embeddings.device
    assert values.dtype == embeddings.dtype
    assert torch.count_nonzero(values) == 0


@pytest.mark.parametrize(
    ("module_path", "expected_runners", "expected_lr"),
    [
        (
            "experiments.two_factor_reward_state_REINFORCE_cycle_4."
            "reward_factor_1_context32_l3.experiment",
            4,
            4.2e-4,
        ),
        (
            "experiments.two_factor_reward_state_REINFORCE_cycle_4."
            "reward_factor_1_context32_l3_small_batch.experiment",
            4,
            2e-4,
        ),
        (
            "experiments.two_factor_reward_state_REINFORCE_cycle_4."
            "reward_both_context32_l3.experiment",
            4,
            4.2e-4,
        ),
    ],
)
def test_cycle_4_context32_l3_arms(tmp_path, module_path, expected_runners, expected_lr):
    module = importlib.import_module(module_path)
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090"],
    )
    config = module.build_config(context)
    spec = config.get_rl_module_spec()

    assert spec.model_config["context_len"] == 32
    assert spec.model_config["n_layers"] == 3
    assert spec.model_config["d_model"] == 64
    if "small_batch" in module_path:
        assert config.train_batch_size_per_learner == 32_768
        assert config.lr == 2e-4
    else:
        assert config.train_batch_size_per_learner == 32_768
        assert config.lr == 4.2e-4
    resolved_runners = (
        expected_runners
        if expected_runners == 4
        else resolve_env_runners(context.hardware, expected_runners)
    )
    assert config.num_env_runners == resolved_runners
    assert config.lr == expected_lr
    config.validate()


@pytest.mark.parametrize(
    ("module_path", "expected_layers", "expected_temp"),
    [
        (
            "experiments.two_factor_reward_state_REINFORCE_cycle_4."
            "reward_both_context32_l4.experiment",
            4,
            1.0,
        ),
        (
            "experiments.two_factor_reward_state_REINFORCE_cycle_4."
            "reward_both_context32_l3_sampling_temp.experiment",
            3,
            1.5,
        ),
    ],
)
def test_cycle_4_reward_both_context32_variants(
    tmp_path,
    module_path,
    expected_layers,
    expected_temp,
):
    module = importlib.import_module(module_path)
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090"],
    )
    config = module.build_config(context)
    spec = config.get_rl_module_spec()

    assert spec.model_config["context_len"] == 32
    assert spec.model_config["n_layers"] == expected_layers
    assert float(spec.model_config.get("sampling_temperature", 1.0)) == expected_temp
    assert config.num_env_runners == 4
    assert config.lr == 4.2e-4
    config.validate()


def test_cycle_4_continue_30m_writes_5m_checkpoint_spec(tmp_path):
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
        CONTINUATION_SPEC_FILENAME,
        STEP_CHECKPOINT_INTERVAL_5M,
        _resolve_step_checkpoint_interval,
    )

    module = importlib.import_module(
        "experiments.two_factor_reward_state_REINFORCE_cycle_4."
        "reward_both_context32_l3_continue_30m.experiment"
    )
    checkpoint = tmp_path / "prior" / "steps_030000000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "rllib_checkpoint.json").write_text("{}")
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results" / "continue",
        artifacts_dir=tmp_path / "artifacts" / "continue",
        seed=42,
        smoke=False,
        resume_from=checkpoint,
        hardware=PROFILES["cuda4090"],
    )
    with pytest.raises(RuntimeError):
        module.run(context)
    spec_path = context.artifacts_dir / CONTINUATION_SPEC_FILENAME
    assert spec_path.is_file()
    payload = json.loads(spec_path.read_text())
    assert payload["target_agent_steps"] == 60_000_000
    assert payload["step_checkpoint_interval"] == STEP_CHECKPOINT_INTERVAL_5M
    assert _resolve_step_checkpoint_interval(context) == STEP_CHECKPOINT_INTERVAL_5M


def test_training_tracking_occupancy_from_return_over_length():
    from experiments.two_factor_reward_state_REINFORCE_cycle_4.training_tracking import (
        occupancy_fraction_from_metrics,
    )

    metrics = {
        "env_runners": {
            "episode_return_mean": 610.0,
            "episode_len_mean": 1024.0,
            "num_env_steps_sampled_lifetime": 30_000_000.0,
        },
        "training_iteration": 900,
    }
    fraction = occupancy_fraction_from_metrics(metrics, condition="reward_both")
    assert fraction == pytest.approx(610.0 / 1024.0)
    assert 100.0 * fraction == pytest.approx(100.0 * 610.0 / 1024.0)
