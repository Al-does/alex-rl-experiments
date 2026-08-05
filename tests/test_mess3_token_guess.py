"""Focused tests for the rewarded MESS3 token-prediction study."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import torch

import pytest

from envs.hmm import HMMEnv, stationary_distribution
from envs.mess3.model import PASSIVE_TRANSITION_MATRIX, emission_matrix
from experiments.mess3_belief_geometry_2026_07.probe import ProbeData
from experiments.mess3_token_guess_cycle_1.analysis import (
    PROBE_RANK,
    fit_reduced_rank_affine,
)
from experiments.mess3_token_guess_cycle_1.baselines import (
    calibrate,
    expected_accuracy_band,
    trivial_feature_r2,
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


def _passive_probe_data(n_steps: int, *, seed: int, policy: str) -> ProbeData:
    """Simulate the delay-one decision stream with exact predictive beliefs."""

    transition = np.asarray(PASSIVE_TRANSITION_MATRIX)
    emission = emission_matrix(0.85)
    initial = stationary_distribution(transition)
    rng = np.random.default_rng(seed)

    beliefs, tokens, previous, states, actions, rewards = [], [], [], [], [], []
    state = rng.choice(3, p=initial)
    belief = initial.copy()
    visible, older = 0, 0
    for step in range(n_steps + 1):
        token = rng.choice(3, p=emission[state])
        if step > 0:
            predictive = belief @ emission
            action = int(
                predictive.argmax() if policy == "bayes" else visible
            )
            beliefs.append(belief.copy())
            tokens.append(visible)
            previous.append(older)
            states.append(int(state))
            actions.append(action)
            rewards.append(float(action == token))
        posterior = belief * emission[:, token]
        belief = (posterior / posterior.sum()) @ transition
        state = rng.choice(3, p=transition[state])
        visible, older = token, visible

    belief_array = np.asarray(beliefs)
    return ProbeData(
        activations=belief_array,
        beliefs=belief_array,
        diagnostic_beliefs=belief_array,
        tokens=np.asarray(tokens, dtype=np.int64),
        previous_tokens=np.asarray(previous, dtype=np.int64),
        states=np.asarray(states, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.float64).reshape(-1, 1),
        rewards=np.asarray(rewards, dtype=np.float64),
    )


def test_accuracy_band_places_echo_and_bayes_policies_at_the_range_ends():
    emission = emission_matrix(0.85)
    echo = expected_accuracy_band(
        _passive_probe_data(20_000, seed=3, policy="echo"),
        emission,
    )
    bayes = expected_accuracy_band(
        _passive_probe_data(20_000, seed=3, policy="bayes"),
        emission,
    )

    assert 0.66 < echo["expected_accuracy_echo_last_token"] < 0.68
    assert 0.68 < echo["expected_accuracy_bayes"] < 0.70
    assert 0.010 < echo["accuracy_range"] < 0.025
    assert echo["accuracy_fraction_of_range"] == pytest.approx(0.0, abs=1e-9)
    assert bayes["accuracy_fraction_of_range"] == pytest.approx(1.0, abs=1e-9)


def test_accuracy_band_rejects_decisions_taken_before_any_token_is_visible():
    data = _passive_probe_data(64, seed=5, policy="echo")
    blinded = np.array(data.tokens)
    blinded[0] = -1
    with pytest.raises(ValueError, match="visible token"):
        expected_accuracy_band(
            replace(data, tokens=blinded),
            emission_matrix(0.85),
        )


def test_recent_token_features_already_explain_most_of_the_belief_simplex():
    train = _passive_probe_data(20_000, seed=7, policy="bayes")
    test = _passive_probe_data(20_000, seed=8, policy="bayes")
    scores = trivial_feature_r2(
        train,
        test,
        n_tokens=3,
        fit=fit_reduced_rank_affine,
        rank=PROBE_RANK,
    )

    assert 0.75 < scores["r_squared_last_token"] < 0.85
    assert 0.88 < scores["r_squared_last_two_tokens"] < 0.96
    assert 0.84 < scores["r_squared_own_action_only"] < 0.92
    assert (
        scores["r_squared_last_two_tokens"] > scores["r_squared_last_token"]
    )


def test_calibration_marks_a_perfect_probe_as_beating_the_trivial_floors():
    train = _passive_probe_data(8_000, seed=9, policy="bayes")
    test = _passive_probe_data(8_000, seed=10, policy="bayes")
    weight, bias = fit_reduced_rank_affine(
        train.activations,
        train.beliefs,
        rank=PROBE_RANK,
    )
    predicted = test.activations @ weight + bias
    metrics = calibrate(
        train,
        test,
        predicted,
        emission_matrix=emission_matrix(0.85),
        fit=fit_reduced_rank_affine,
        rank=PROBE_RANK,
    )

    for depth in (1, 2):
        assert metrics[f"r_squared_within_branch_depth{depth}"] > 0.99
    assert metrics["r_squared_last_token"] < 0.9
    assert metrics["accuracy_fraction_of_range"] == pytest.approx(1.0, abs=1e-9)


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
