from __future__ import annotations

import numpy as np

from envs.gol.model import controlled_kernels, gol_model
from experiments.gol_reward_state_action_symmetry_cycle_1_entropy_0_01_20m.posthoc_analysis import (
    categorical_prediction,
    joint_outcome_features,
    replay_beliefs,
)


def test_joint_outcomes_are_normalized_and_match_separate_marginals():
    rng = np.random.default_rng(7)
    beliefs = rng.dirichlet(np.ones(4), size=12)
    kernels = controlled_kernels(variant=3, speed="half")
    outcomes = joint_outcome_features(beliefs, kernels).reshape(-1, 4, 2, 2)
    np.testing.assert_allclose(outcomes.sum(axis=(2, 3)), 1.0)
    np.testing.assert_allclose(
        outcomes.sum(axis=3),
        np.einsum("ns,atsd->nat", beliefs, kernels),
    )
    np.testing.assert_allclose(
        outcomes[..., 1].sum(axis=2),
        beliefs @ kernels.sum(axis=1)[:, :, 2].T,
    )


def test_belief_replay_uses_current_token_and_previous_action():
    variant = 3
    kernels = controlled_kernels(variant=variant, speed="half")
    histories = np.zeros((4, 2, 6), dtype=np.float32)
    choices = [
        ((0, 1), (1, 3)),
        ((1, 2), (0, 0)),
        ((0, 3), (1, 1)),
    ]
    for step, row in enumerate(choices, start=1):
        for environment, (token, action) in enumerate(row):
            histories[step, environment, token] = 1
            histories[step, environment, 2 + action] = 1
    replayed = replay_beliefs(histories, variant=variant, speed="half")
    expected = np.empty_like(replayed)
    expected[0] = gol_model(variant=variant, speed="half").initial_distribution
    for step, row in enumerate(choices, start=1):
        for environment, (token, action) in enumerate(row):
            update = expected[step - 1, environment] @ kernels[action, token]
            expected[step, environment] = update / update.sum()
    np.testing.assert_allclose(replayed, expected)


def test_categorical_prediction_uses_training_centroids_and_mean_fallback():
    train_keys = np.array([[1, 2], [1, 2], [2, 3]])
    test_keys = np.array([[1, 2], [9, 9]])
    targets = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    prediction = categorical_prediction(train_keys, test_keys, targets)
    np.testing.assert_allclose(prediction[0], [0.5, 0.5])
    np.testing.assert_allclose(prediction[1], targets.mean(axis=0))
