"""Focused tests for the rewarded MESS3 token-prediction study."""

from __future__ import annotations

import math

import numpy as np
import torch

from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_1.analysis import (
    fit_reduced_rank_affine,
)
from experiments.mess3_token_guess_cycle_1.comparison.experiment import (
    ARMS,
    ENV_CONFIG,
    build_config,
)
from experiments.mess3_token_guess_cycle_1.entropy_reward import (
    EntropyRewardPPOTorchLearner,
    add_categorical_entropy_reward,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def test_token_guess_task_has_one_action_per_token_and_scores_next_reveal():
    config = {
        **ENV_CONFIG,
        "episode_length": 3,
        "diagnostics": {
            "tokens": True,
            "rewards": True,
            "transitions": True,
        },
    }
    environment = HMMEnv(config)
    try:
        observation, info = environment.reset(seed=7)
        assert environment.action_space.n == environment.model.n_tokens == 3
        np.testing.assert_array_equal(observation, np.zeros(3))

        privileged_next_token = info["raw_token_current"]
        next_observation, reward, _, _, step_info = environment.step(
            privileged_next_token
        )
        assert reward == 1.0
        assert step_info["raw_token_before"] == privileged_next_token
        assert step_info["reward_components"]["token_guess_correct"] == 1.0
        assert next_observation[privileged_next_token] == 1.0

        wrong_guess = (step_info["raw_token_current"] + 1) % 3
        _, reward, _, _, step_info = environment.step(wrong_guess)
        assert reward == 0.0
        assert step_info["reward_components"]["token_guess_correct"] == 0.0
    finally:
        environment.close()


def test_entropy_reward_is_added_before_gae_without_rewarding_padding():
    rewards = torch.tensor([[1.0, 0.0, 0.0]])
    logits = torch.zeros((1, 3, 3))
    valid = torch.tensor([[True, True, False]])
    augmented, bonus = add_categorical_entropy_reward(
        rewards,
        logits,
        coefficient=0.05,
        valid=valid,
    )
    expected = 0.05 * math.log(3.0)
    torch.testing.assert_close(
        bonus,
        torch.tensor([[expected, expected, 0.0]]),
    )
    torch.testing.assert_close(augmented, rewards + bonus)


def test_rank_two_affine_least_squares_recovers_held_out_map():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(600, 8))
    basis = rng.normal(size=(8, 2))
    simplex_coordinates = features @ basis
    target_weight = np.array(
        [
            [0.2, -0.1, -0.1],
            [-0.05, 0.15, -0.1],
        ]
    )
    targets = simplex_coordinates @ target_weight + np.array([0.3, 0.4, 0.3])

    weight, bias = fit_reduced_rank_affine(
        features[:400],
        targets[:400],
        rank=2,
    )
    predicted = features[400:] @ weight + bias
    np.testing.assert_allclose(predicted, targets[400:], atol=1e-10)
    assert np.linalg.matrix_rank(weight, tol=1e-10) == 2


def test_all_comparison_arms_build_fresh_controlled_smoke_configs(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    configs = {arm.name: build_config(context, arm.name) for arm in ARMS}

    for config in configs.values():
        assert config.seed == 42
        assert config.num_env_runners == 0
        assert config.train_batch_size_per_learner == 2_048
        assert config.entropy_coeff == 0.0
        assert config.env_config == ENV_CONFIG

    assert configs["predictive_loss"].learner_class is not None
    assert (
        configs["max_entropy"].learner_class
        is EntropyRewardPPOTorchLearner
    )
