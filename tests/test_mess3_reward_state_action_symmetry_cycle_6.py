"""Focused recipe tests for the REINFORCE action-symmetry cycle."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_6.design import (
    CYCLE_6_TRANSITION_MATRIX,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    BASE_MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    TOTAL_ENV_STEPS,
    ReinforceTransformerModel,
    environment_config,
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


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_reinforce_variants_build_fresh_monte_carlo_configs(
    smoke_context,
    variant,
):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_6."
        f"variant_{variant}.experiment"
    )

    first = module.build_config(smoke_context)
    second = module.build_config(smoke_context)
    spec = first.get_rl_module_spec()

    assert first is not second
    assert first.env_config["task"]["kwargs"]["variant"] == variant
    assert first.gamma == 0.99
    assert first.lambda_ == 1.0
    assert first.use_critic is False
    assert first.use_gae is False
    assert first.use_kl_loss is False
    assert first.vf_loss_coeff == 0.0
    assert first.entropy_coeff == 0.0
    assert first.num_epochs == 1
    assert first.minibatch_size is None
    assert first.batch_mode == "complete_episodes"
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert spec.module_class is ReinforceTransformerModel
    assert spec.model_config["d_model"] == 64
    assert spec.model_config["n_layers"] == 4
    assert spec.model_config["n_heads"] == 1
    assert spec.model_config["context_len"] == 10
    first.validate()

    environment = HMMEnv(first.env_config)
    try:
        assert environment.action_space.n == 3
        np.testing.assert_array_equal(
            environment.model.transition_matrix,
            CYCLE_6_TRANSITION_MATRIX,
        )
    finally:
        environment.close()


def test_cycle_6_budget_and_model_match_requested_recipe():
    assert TOTAL_ENV_STEPS == 8_000_000
    assert BASE_MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "context_len": 10,
        "max_seq_len": 32,
        "grad_checkpointing": False,
    }


@pytest.mark.parametrize(
    ("variant", "condition", "temperature", "entropy_coeff"),
    (
        (2, "ctx32", 1.0, 0.0),
        (3, "t15_ctx32", 1.5, 0.0),
        (2, "ctx32_ent", 1.0, [[0, 0.03], [1, 0.0]]),
    ),
)
def test_ctx32_arms_change_only_temperature_and_context(
    smoke_context, variant, condition, temperature, entropy_coeff
):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_6."
        f"variant_{variant}_{condition}.experiment"
    )

    config = module.build_config(smoke_context)
    spec = config.get_rl_module_spec()

    assert config.env_config["task"]["kwargs"]["variant"] == variant
    assert spec.module_class is ReinforceTransformerModel
    assert spec.model_config["sampling_temperature"] == temperature
    assert spec.model_config["context_len"] == 32
    assert spec.model_config["grad_checkpointing"] is True
    for key in ("d_model", "n_layers", "n_heads", "max_seq_len"):
        assert spec.model_config[key] == BASE_MODEL_CONFIG[key]
    assert config.lr == 4.2e-4
    assert config.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert config.gamma == 0.99
    assert config.lambda_ == 1.0
    assert config.entropy_coeff == entropy_coeff
    assert config.use_critic is False
    config.validate()


def test_ctx32_ent_schedule_reaches_zero_1m_before_target(tmp_path):
    from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
        write_budget_spec,
    )

    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_6."
        "variant_2_ctx32_ent.experiment"
    )
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )
    write_budget_spec(context, 10_000_000)

    assert module.entropy_schedule(context) == [[0, 0.03], [9_000_000, 0.0]]


def test_t15_ctx32_logits_divided_in_rollout_and_train(tmp_path):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_6."
        "variant_3_t15_ctx32.experiment"
    )
    environment = HMMEnv(environment_config(2))
    try:
        torch.manual_seed(11)
        policy = ReinforceTransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config=dict(module.T15_CTX32_MODEL_CONFIG),
        ).eval()
    finally:
        environment.close()

    embeddings = torch.randn(2, 3, 64)
    expected = (
        policy.heads.action_distribution_inputs(embeddings) / 1.5
    )
    for training in (False, True):
        outputs = policy._outputs(embeddings, None, training=training)
        torch.testing.assert_close(
            outputs[Columns.ACTION_DIST_INPUTS], expected
        )
    torch.testing.assert_close(
        policy.action_distribution_inputs(embeddings), expected
    )

    batch = {
        Columns.OBS: torch.randn(1, 2, policy._obs_dim),
        Columns.STATE_IN: {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in policy.get_initial_state().items()
        },
    }
    train = policy.forward_train(batch)
    rollout = policy.forward_exploration(batch)
    torch.testing.assert_close(
        train[Columns.ACTION_DIST_INPUTS],
        rollout[Columns.ACTION_DIST_INPUTS],
    )
    assert policy.sequence_lookback == 128


def test_reinforce_model_supplies_device_native_zero_baseline():
    embeddings = torch.randn(2, 7, 64)
    values = ReinforceTransformerModel.compute_values(
        object(),
        {},
        embeddings=embeddings,
    )

    assert values.shape == (2, 7)
    assert values.device == embeddings.device
    assert values.dtype == embeddings.dtype
    assert torch.count_nonzero(values) == 0


class _RecordingAlgorithm:
    def __init__(self):
        self.saved = []

    def save_to_path(self, path):
        self.saved.append(Path(path))
        Path(path).mkdir(parents=True)
        return str(path)


def _context(tmp_path, **overrides):
    values = {
        "experiment_dir": tmp_path,
        "results_dir": tmp_path / "results",
        "artifacts_dir": tmp_path / "artifacts",
        "seed": 42,
    }
    values.update(overrides)
    return RunContext(**values)


def _result(steps, iteration):
    return {
        "env_runners": {"num_env_steps_sampled_lifetime": steps},
        "training_iteration": iteration,
    }


def test_step_checkpoint_callback_saves_only_on_threshold_crossing(tmp_path):
    from functools import partial

    from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
        _save_step_checkpoint_and_upload,
    )

    context = _context(tmp_path)
    algorithm = _RecordingAlgorithm()
    callback = partial(
        _save_step_checkpoint_and_upload,
        checkpoint_root=str(tmp_path / "step_checkpoints"),
        step_interval=25_000_000,
        context=context,
        upload=False,
    )

    callback(algorithm=algorithm, result=_result(8_226_448, 21))
    assert algorithm.saved == []

    callback(algorithm=algorithm, result=_result(25_100_000, 64))
    assert [p.name for p in algorithm.saved] == ["steps_025100000"]

    callback(algorithm=algorithm, result=_result(30_000_000, 77))
    assert len(algorithm.saved) == 1

    callback(algorithm=algorithm, result=_result(51_000_000, 130))
    assert [p.name for p in algorithm.saved] == [
        "steps_025100000",
        "steps_051000000",
    ]

    index = json.loads((tmp_path / "step_checkpoints/index.json").read_text())
    assert [c["agent_steps"] for c in index["checkpoints"]] == [
        25_100_000,
        51_000_000,
    ]
    assert index["next_threshold"] == 75_000_000

    uploads = (
        tmp_path / "results" / "checkpoint_uploads.jsonl"
    ).read_text().splitlines()
    assert len(uploads) == 2
    assert json.loads(uploads[0])["uploaded"] is False


def test_continuation_step_target_defaults_to_300m(tmp_path):
    from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
        _continuation_step_target,
    )

    context = _context(tmp_path)
    assert _continuation_step_target(context) == 300_000_000

    context.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (context.artifacts_dir / "budget_spec.json").write_text(
        json.dumps({"target_agent_steps": 150_000_000})
    )
    assert _continuation_step_target(context) == 150_000_000
