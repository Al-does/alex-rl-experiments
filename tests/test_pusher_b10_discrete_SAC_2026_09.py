from __future__ import annotations

from dataclasses import replace
import importlib

import pytest
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.pusher_b.process import CONTEXT_LENGTH, PRESETS, TOKEN_COUNT
from experiments.pusher_b10_discrete_SAC_2026_09.model import (
    PusherSharedTrunkSAC,
    PusherSplitSAC,
    SharedTrunkSACTorchLearner,
    trainable_parameter_count,
)
from experiments.pusher_b10_discrete_SAC_2026_09.shared import (
    ACTOR_LEARNING_RATE,
    ALPHA_LEARNING_RATE,
    CHECKPOINT_INTERVAL_ENV_STEPS,
    CRITIC_LEARNING_RATE,
    LEARNER_MINIBATCH_COUNT,
    LEARNER_MINIBATCH_SIZE,
    LEARNING_STARTS,
    LOG_CHECKPOINT_END_ENV_STEPS,
    MODEL_CONFIG,
    REPLAY_CAPACITY,
    SHARED_ENCODER_LEARNING_RATE,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_LEARNING_STARTS,
    SMOKE_REPLAY_CAPACITY,
    TARGET_ENTROPY,
    TOTAL_ENV_STEPS,
    TRAIN_BATCH_SIZE,
    build_config,
    checkpoint_decision,
    resolved_recipe,
    sac_environment_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path, *, smoke: bool = True) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=smoke,
        hardware=PROFILES["cpu"],
    )


def _build_module(config, env: HMMEnv):
    spec = replace(
        config.rl_module_spec,
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_config={
            **config.rl_module_spec.model_config,
            "twin_q": config.twin_q,
        },
    )
    assert isinstance(spec, RLModuleSpec)
    return spec.build()


def test_environment_is_b10_with_a_complete_fixed_token_history():
    config = sac_environment_config()
    assert config["model"]["factory"].endswith("pusher_b_model_b10")
    assert config["observation"] == {
        "token": {"depth": CONTEXT_LENGTH},
        "action": None,
    }
    assert config["episode_length"] == 127
    env = HMMEnv(config)
    try:
        assert env.observation_space.shape == (
            CONTEXT_LENGTH * TOKEN_COUNT,
        )
        assert env.action_space.n == TOKEN_COUNT == 2
    finally:
        env.close()


@pytest.mark.parametrize(
    ("architecture", "module_class"),
    [
        ("shared_trunk", PusherSharedTrunkSAC),
        ("split_transformers", PusherSplitSAC),
    ],
)
def test_smoke_and_full_configs_match_the_preregistered_recipe(
    tmp_path,
    architecture,
    module_class,
):
    smoke = build_config(_context(tmp_path), architecture=architecture)
    assert smoke is not build_config(
        _context(tmp_path),
        architecture=architecture,
    )
    assert smoke.rl_module_spec.module_class is module_class
    assert smoke.rl_module_spec.model_config == MODEL_CONFIG
    assert smoke.gamma == 0.0
    assert smoke.n_step == 1
    assert smoke.twin_q
    assert smoke.tau == 0.005
    assert smoke.actor_lr == ACTOR_LEARNING_RATE
    assert smoke.critic_lr == CRITIC_LEARNING_RATE
    assert smoke.alpha_lr == ALPHA_LEARNING_RATE
    assert smoke.target_entropy == TARGET_ENTROPY
    assert smoke.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert (
        smoke.num_steps_sampled_before_learning_starts
        == SMOKE_LEARNING_STARTS
    )
    assert smoke.replay_buffer_config["capacity"] == SMOKE_REPLAY_CAPACITY
    assert smoke.num_env_runners == 0
    assert smoke.num_envs_per_env_runner == 1
    assert smoke.rollout_fragment_length == 1

    full_context = _context(tmp_path, smoke=False)
    full = build_config(full_context, architecture=architecture)
    assert full.train_batch_size_per_learner == TRAIN_BATCH_SIZE
    assert (
        full.num_steps_sampled_before_learning_starts == LEARNING_STARTS
    )
    assert full.replay_buffer_config["capacity"] == REPLAY_CAPACITY
    recipe = resolved_recipe(full_context, architecture=architecture)
    assert recipe["parameters"] == PRESETS["b10"]
    assert recipe["total_env_steps"] == TOTAL_ENV_STEPS
    assert recipe["train_batch_size_per_learner"] == TRAIN_BATCH_SIZE
    assert recipe["learner_minibatch_count"] == LEARNER_MINIBATCH_COUNT
    assert recipe["learner_minibatch_size"] == LEARNER_MINIBATCH_SIZE
    assert recipe["checkpoint_schedule"]["phase_boundary_env_steps"] == (
        LOG_CHECKPOINT_END_ENV_STEPS
    )
    assert recipe["checkpoint_schedule"]["fixed_interval_env_steps"] == (
        CHECKPOINT_INTERVAL_ENV_STEPS
    )
    assert recipe["checkpoint_schedule"]["fixed_targets_env_steps"] == [
        55_000_000,
        80_000_000,
        105_000_000,
        130_000_000,
        155_000_000,
        180_000_000,
        205_000_000,
        230_000_000,
        255_000_000,
        280_000_000,
    ]

    if architecture == "shared_trunk":
        assert full.learner_class is SharedTrunkSACTorchLearner
        assert full.learner_config_dict == {
            "shared_encoder_learning_rate": SHARED_ENCODER_LEARNING_RATE
        }
    else:
        assert full.learner_class is SACTorchLearner


def test_shared_and_split_modules_have_the_requested_parameter_topology(
    tmp_path,
):
    env = HMMEnv(sac_environment_config())
    try:
        shared = _build_module(
            build_config(_context(tmp_path), architecture="shared_trunk"),
            env,
        )
        split = _build_module(
            build_config(
                _context(tmp_path),
                architecture="split_transformers",
            ),
            env,
        )
    finally:
        env.close()

    assert shared.pi_encoder is shared.qf_encoder
    assert shared.pi_encoder is shared.qf_twin_encoder
    assert split.pi_encoder is not split.qf_encoder
    assert split.pi_encoder is not split.qf_twin_encoder
    assert split.qf_encoder is not split.qf_twin_encoder
    assert trainable_parameter_count(shared) < trainable_parameter_count(split)


def test_checkpoint_schedule_switches_after_thirty_million_steps():
    records = []
    assert checkpoint_decision(
        training_iteration=1,
        env_steps=1_000_000,
        records=records,
    ) == ("log", 1_000_000)
    assert checkpoint_decision(
        training_iteration=3,
        env_steps=3_000_000,
        records=records,
    ) is None
    assert checkpoint_decision(
        training_iteration=32,
        env_steps=29_000_000,
        records=records,
    ) == ("log", 29_000_000)
    assert checkpoint_decision(
        training_iteration=64,
        env_steps=31_000_000,
        records=records,
    ) == ("interval", 30_000_000)

    records.append(
        {
            "training_iteration": 64,
            "target_env_steps": 30_000_000,
        }
    )
    assert checkpoint_decision(
        training_iteration=65,
        env_steps=54_999_999,
        records=records,
    ) is None
    assert checkpoint_decision(
        training_iteration=66,
        env_steps=55_000_000,
        records=records,
    ) == ("interval", 55_000_000)


@pytest.mark.parametrize(
    "module_name",
    [
        (
            "experiments.pusher_b10_discrete_SAC_2026_09."
            "shared_trunk.experiment"
        ),
        (
            "experiments.pusher_b10_discrete_SAC_2026_09."
            "split_transformers.experiment"
        ),
    ],
)
def test_both_experiment_leaves_are_importable(tmp_path, module_name):
    module = importlib.import_module(module_name)
    assert callable(module.run)
    assert module.build_config(_context(tmp_path)).env_config == (
        sac_environment_config()
    )


def test_smoke_recipe_uses_only_the_smoke_budget(tmp_path):
    recipe = resolved_recipe(
        _context(tmp_path),
        architecture="shared_trunk",
    )
    assert recipe["total_env_steps"] == SMOKE_ENV_STEPS
