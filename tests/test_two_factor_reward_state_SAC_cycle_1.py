"""Scientific, model, and wiring tests for two-factor reward-state SAC."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.sac import SAC
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.two_factor_reward_state_SAC_cycle_1.design import (
    REFERENCE_VALUES,
    demand_audit,
)
from experiments.two_factor_reward_state_SAC_cycle_1.model import (
    FLAT_OBSERVATION_WIDTH,
    TwoFactorRewardSAC,
    TwoFactorSACEncoder,
)
from experiments.two_factor_reward_state_SAC_cycle_1.process import (
    CONTEXT_LENGTH,
    JOINT_TOKEN_COUNT,
    MESS3_ALPHA,
    TRANSITION_MATRIX,
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_1.shared import (
    EightMinibatchSAC,
    LEARNER_MINIBATCH_COUNT,
    LEARNER_MINIBATCH_SIZE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_LEARNING_STARTS,
    TARGET_ENTROPY,
    TARGET_ENTROPY_FRACTION,
    TOTAL_ENV_STEPS,
    TRAIN_BATCH_SIZE,
    TRAINING_INTENSITY,
    _resolved_recipe,
)
from experiments.two_factor_reward_state_SAC_cycle_1.task import (
    ACTION_PAIRS,
    CONDITIONS,
    N_ACTIONS,
    joint_transition,
    shifted_transition,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def _module() -> TwoFactorRewardSAC:
    module = TwoFactorRewardSAC(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(FLAT_OBSERVATION_WIDTH,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(N_ACTIONS),
        model_config={**MODEL_CONFIG, "twin_q": True},
        inference_only=False,
    )
    module.make_target_networks()
    return module


def test_pretraining_audit_matches_reference_and_passes_required_gate():
    report = demand_audit()

    assert report["fully_observed"] == pytest.approx(
        REFERENCE_VALUES["fully_observed"],
        abs=1e-12,
    )
    assert report["qmdp"] == pytest.approx(REFERENCE_VALUES["qmdp"], abs=5e-4)
    assert report["best_constant"] == pytest.approx(
        REFERENCE_VALUES["best_constant"],
        abs=1e-12,
    )
    assert report["reactive"] == pytest.approx(
        REFERENCE_VALUES["reactive"],
        abs=5e-4,
    )
    assert report["demand_gap"] >= 0.015
    assert report["demand_gap_standard_error"] <= 5e-4


def test_flat_action_order_and_destination_rotations_are_exact():
    assert ACTION_PAIRS == (
        (0, 0),
        (1, 1),
        (1, 2),
        (0, 1),
        (0, 2),
        (2, 1),
        (2, 2),
        (1, 0),
        (2, 0),
    )
    for shift in range(3):
        controlled = shifted_transition(shift)
        for old_destination in range(3):
            np.testing.assert_array_equal(
                controlled[:, (old_destination + shift) % 3],
                TRANSITION_MATRIX[:, old_destination],
            )
    for action, (first, second) in enumerate(ACTION_PAIRS):
        np.testing.assert_allclose(
            joint_transition(action),
            np.kron(shifted_transition(first), shifted_transition(second)),
        )


@pytest.mark.parametrize(
    ("condition", "joint_state", "expected"),
    [
        ("reward_both", 8, 2.0),
        ("reward_both", 6, 1.0),
        ("reward_factor_1", 6, 1.0),
        ("reward_factor_1", 2, 0.0),
        ("reward_factor_2", 2, 1.0),
        ("reward_factor_2", 6, 0.0),
    ],
)
def test_reward_arms_select_the_requested_state_two_factors(
    condition,
    joint_state,
    expected,
):
    environment = HMMEnv(environment_config(condition))
    try:
        task = environment.task
        from envs.hmm import ActionDecision, TransitionEvent

        reward, components = task.reward(
            TransitionEvent(
                step=0,
                state_before=joint_state,
                state_after=joint_state,
                raw_token_before=0,
                raw_token_after=0,
            ),
            ActionDecision(0, 0, joint_transition(0)),
        )
        assert reward == expected
        assert set(components) == {
            "factor_1_occupancy_reward",
            "factor_2_occupancy_reward",
        }
    finally:
        environment.close()


def test_environment_exposes_only_joint_tokens_and_flat_actions():
    config = environment_config("reward_both")
    config["diagnostics"] = {"belief": True, "tokens": True, "transitions": True}
    environment = HMMEnv(config)
    try:
        observation, info = environment.reset(seed=9)
        assert MESS3_ALPHA == 0.55
        assert environment.model.n_states == 9
        assert environment.model.n_tokens == JOINT_TOKEN_COUNT == 9
        assert environment.action_space == gym.spaces.Discrete(9)
        assert environment.observation_space.shape == (FLAT_OBSERVATION_WIDTH,)
        assert observation[:JOINT_TOKEN_COUNT].sum() == 1.0
        assert observation[JOINT_TOKEN_COUNT : CONTEXT_LENGTH * 9].sum() == 0.0
        assert observation[CONTEXT_LENGTH * 9 :].sum() == 0.0
        assert "state_current" not in info

        next_observation, _, _, _, next_info = environment.step(2)
        np.testing.assert_allclose(
            next_info["executed_transition_matrix"],
            joint_transition(2),
        )
        action_history = next_observation[CONTEXT_LENGTH * 9 :].reshape(
            CONTEXT_LENGTH,
            9,
        )
        assert action_history[0, 2] == 1.0
        assert action_history.sum() == 1.0
    finally:
        environment.close()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_each_leaf_builds_a_fresh_twenty_million_step_sac_recipe(
    tmp_path,
    condition,
):
    module = importlib.import_module(
        "experiments.two_factor_reward_state_SAC_cycle_1."
        f"{condition}.experiment"
    )
    context = _context(tmp_path)
    first = module.build_config(context)
    second = module.build_config(context)

    assert first is not second
    assert first.seed == 42
    assert first.gamma == 0.99
    assert first.n_step == 1
    assert first.twin_q is True
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.num_steps_sampled_before_learning_starts == SMOKE_LEARNING_STARTS
    assert first.num_env_runners == 0
    assert first.env_config["task"]["kwargs"]["condition"] == condition
    assert first.rl_module_spec.module_class is TwoFactorRewardSAC
    assert TOTAL_ENV_STEPS == 5_000_000


def test_cuda_recipe_uses_profiled_eager_batched_replay(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090_gpuinfer"],
    )
    config = importlib.import_module(
        "experiments.two_factor_reward_state_SAC_cycle_1."
        "reward_both.experiment"
    ).build_config(context)

    assert config.algo_class is EightMinibatchSAC
    assert config.train_batch_size_per_learner == TRAIN_BATCH_SIZE == 8_192
    assert LEARNER_MINIBATCH_COUNT == 8
    assert LEARNER_MINIBATCH_SIZE == 1_024
    assert config.training_intensity == TRAINING_INTENSITY == 1.0
    assert config.target_entropy == pytest.approx(0.6 * math.log(9))
    assert config.target_entropy == TARGET_ENTROPY
    assert config.rollout_fragment_length == CONTEXT_LENGTH == 64
    assert config.min_sample_timesteps_per_iteration == TRAIN_BATCH_SIZE
    assert config.torch_compile_learner is False


@pytest.mark.parametrize(
    ("total_batch_size", "expected_minibatch_size"),
    [(TRAIN_BATCH_SIZE, LEARNER_MINIBATCH_SIZE), (SMOKE_BATCH_SIZE, SMOKE_BATCH_SIZE)],
)
def test_custom_sac_passes_gpu_sized_minibatches_to_learner(
    monkeypatch,
    total_batch_size,
    expected_minibatch_size,
):
    calls = []

    class LearnerGroup:
        def update(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"updated": True}

    def parent_training_step(self):
        self.learner_group.update(episodes=["episode"])
        return {"complete": True}

    monkeypatch.setattr(SAC, "training_step", parent_training_step)
    algorithm = object.__new__(EightMinibatchSAC)
    algorithm.learner_group = LearnerGroup()
    algorithm.config = SimpleNamespace(total_train_batch_size=total_batch_size)
    original_update = algorithm.learner_group.update

    assert algorithm.training_step() == {"complete": True}
    assert calls == [
        (
            (),
            {
                "episodes": ["episode"],
                "num_epochs": 1,
                "minibatch_size": expected_minibatch_size,
            },
        )
    ]
    assert algorithm.learner_group.update == original_update


def test_resolved_recipe_records_throughput_and_entropy_choices(tmp_path):
    recipe = _resolved_recipe(_context(tmp_path), "reward_both", {})

    assert recipe["train_batch_size_per_learner"] == TRAIN_BATCH_SIZE
    assert recipe["learner_minibatch_count"] == LEARNER_MINIBATCH_COUNT
    assert recipe["learner_minibatch_size"] == LEARNER_MINIBATCH_SIZE
    assert recipe["learner_num_epochs"] == 1
    assert recipe["training_intensity"] == TRAINING_INTENSITY
    assert recipe["target_entropy_fraction_of_categorical_maximum"] == (
        TARGET_ENTROPY_FRACTION
    )
    assert recipe["target_entropy"] == TARGET_ENTROPY
    assert recipe["rollout_fragment_length"] == CONTEXT_LENGTH
    assert recipe["torch_compile_learner"] is False
    assert recipe["min_sample_timesteps_per_iteration"] == TRAIN_BATCH_SIZE


def test_cuda_recipe_uses_profiled_eager_batched_replay(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090_gpuinfer"],
    )
    config = importlib.import_module(
        "experiments.two_factor_reward_state_SAC_cycle_1."
        "reward_both.experiment"
    ).build_config(context)

    assert config.algo_class is EightMinibatchSAC
    assert config.train_batch_size_per_learner == TRAIN_BATCH_SIZE == 8_192
    assert LEARNER_MINIBATCH_COUNT == 8
    assert LEARNER_MINIBATCH_SIZE == 1_024
    assert config.training_intensity == TRAINING_INTENSITY == 1.0
    assert config.target_entropy == pytest.approx(0.6 * math.log(9))
    assert config.target_entropy == TARGET_ENTROPY
    assert config.rollout_fragment_length == CONTEXT_LENGTH == 64
    assert config.min_sample_timesteps_per_iteration == TRAIN_BATCH_SIZE
    assert config.torch_compile_learner is False


@pytest.mark.parametrize(
    ("total_batch_size", "expected_minibatch_size"),
    [(TRAIN_BATCH_SIZE, LEARNER_MINIBATCH_SIZE), (SMOKE_BATCH_SIZE, SMOKE_BATCH_SIZE)],
)
def test_custom_sac_passes_gpu_sized_minibatches_to_learner(
    monkeypatch,
    total_batch_size,
    expected_minibatch_size,
):
    calls = []

    class LearnerGroup:
        def update(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"updated": True}

    def parent_training_step(self):
        self.learner_group.update(episodes=["episode"])
        return {"complete": True}

    monkeypatch.setattr(SAC, "training_step", parent_training_step)
    algorithm = object.__new__(EightMinibatchSAC)
    algorithm.learner_group = LearnerGroup()
    algorithm.config = SimpleNamespace(total_train_batch_size=total_batch_size)
    original_update = algorithm.learner_group.update

    assert algorithm.training_step() == {"complete": True}
    assert calls == [
        (
            (),
            {
                "episodes": ["episode"],
                "num_epochs": 1,
                "minibatch_size": expected_minibatch_size,
            },
        )
    ]
    assert algorithm.learner_group.update == original_update


def test_resolved_recipe_records_throughput_and_entropy_choices(tmp_path):
    recipe = _resolved_recipe(_context(tmp_path), "reward_both", {})

    assert recipe["train_batch_size_per_learner"] == TRAIN_BATCH_SIZE
    assert recipe["learner_minibatch_count"] == LEARNER_MINIBATCH_COUNT
    assert recipe["learner_minibatch_size"] == LEARNER_MINIBATCH_SIZE
    assert recipe["learner_num_epochs"] == 1
    assert recipe["training_intensity"] == TRAINING_INTENSITY
    assert recipe["target_entropy_fraction_of_categorical_maximum"] == (
        TARGET_ENTROPY_FRACTION
    )
    assert recipe["target_entropy"] == TARGET_ENTROPY
    assert recipe["rollout_fragment_length"] == CONTEXT_LENGTH
    assert recipe["torch_compile_learner"] is False
    assert recipe["min_sample_timesteps_per_iteration"] == TRAIN_BATCH_SIZE


def test_actor_and_critics_use_separate_action_aware_transformers():
    torch.manual_seed(3)
    module = _module().eval()
    assert isinstance(module.pi_encoder, TwoFactorSACEncoder)
    assert isinstance(module.qf_encoder, TwoFactorSACEncoder)
    assert isinstance(module.qf_twin_encoder, TwoFactorSACEncoder)
    actor_parameters = {id(value) for value in module.pi_encoder.parameters()}
    critic_parameters = {id(value) for value in module.qf_encoder.parameters()}
    twin_parameters = {id(value) for value in module.qf_twin_encoder.parameters()}
    assert actor_parameters.isdisjoint(critic_parameters)
    assert actor_parameters.isdisjoint(twin_parameters)
    assert critic_parameters.isdisjoint(twin_parameters)

    observations = torch.zeros((2, FLAT_OBSERVATION_WIDTH))
    observations[0, 0] = 1.0
    observations[1, 8] = 1.0
    observations[1, CONTEXT_LENGTH * 9 + 2] = 1.0
    hidden = module.actor_hidden(observations)
    inference = module.forward_inference({Columns.OBS: observations})
    torch.testing.assert_close(
        inference[Columns.ACTION_DIST_INPUTS],
        module.pi(module.encoder.final_norm(hidden)),
    )
    assert hidden.shape == (2, 64)
    assert not torch.allclose(hidden[0], hidden[1])
