from __future__ import annotations

import numpy as np
import torch
from torch import nn

from envs.mess3.model import control_model
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_logit_evidence import (
    condition,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_pattern_checkpoint import (
    bootstrap_mean,
    embedding_variant,
    observations,
    pattern_beliefs,
    pattern_contrasts,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.design import (
    CYCLE_6_TRANSITION_MATRIX,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.task import (
    ActionSymmetryTask,
)


def projection_fixture():
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.input_projection = nn.Linear(6, 8)
    return model


def test_embedding_swap_equals_global_input_relabeling_without_mutating_source():
    model = projection_fixture()
    original = model.encoder.input_projection.weight.detach().clone()
    swapped = embedding_variant(model, "swap")
    inputs = torch.tensor([[1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0]], dtype=torch.float32)
    relabeled = inputs[:, [1, 0, 2, 3, 4, 5]]
    torch.testing.assert_close(
        swapped.encoder.input_projection(inputs),
        model.encoder.input_projection(relabeled),
    )
    torch.testing.assert_close(model.encoder.input_projection.weight, original)
    torch.testing.assert_close(swapped.encoder.input_projection.weight[:, 2:], original[:, 2:])


def test_mean_embedding_erases_token_identity_but_preserves_action_columns():
    model = projection_fixture()
    merged = embedding_variant(model, "mean")
    inputs = torch.tensor([[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]], dtype=torch.float32)
    values = merged.encoder.input_projection(inputs)
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(
        merged.encoder.input_projection.weight[:, 2:],
        model.encoder.input_projection.weight[:, 2:],
    )


def test_factorial_contrast_separates_patterns_from_label_main_effects():
    current_A = np.array([[1, 0, 1, 0]], dtype=float)
    previous_A = np.array([[1, 1, 0, 0]], dtype=float)
    alternating = np.array([[0, 1, 1, 0]], dtype=float)
    values = 10 + 3 * current_A + 5 * previous_A + 7 * alternating
    contrasts = pattern_contrasts(values)
    np.testing.assert_allclose(contrasts["alternating_minus_repeated"], [7])
    np.testing.assert_allclose(contrasts["current_A_minus_B"], [3])
    np.testing.assert_allclose(contrasts["previous_A_minus_B"], [5])


def test_pattern_targets_replay_shared_prefix_and_previous_executed_action():
    model = control_model(alpha=0.85, transition_matrix=CYCLE_6_TRANSITION_MATRIX)
    task = ActionSymmetryTask(model=model, variant=2)
    horizon = 129
    tokens = np.arange(horizon) % 2
    actions = np.ones(horizon, dtype=int)
    actions[0] = -1
    prior = model.initial_distribution[None, :]
    probabilities = []
    prior_at_127 = None
    for step in range(horizon):
        if step == 127:
            prior_at_127 = prior.copy()
        posterior = condition(prior, model.emission_matrix, tokens[step:step + 1])
        probabilities.append(posterior[0, 2])
        prior = posterior @ task.transition_matrix_for_action(1)
    grid = {
        "token": tokens[None, :], "previous_action": actions[None, :],
        "episode": np.zeros((1, horizon), dtype=int),
        "step": np.arange(horizon)[None, :],
        "rho": np.asarray(probabilities)[None, :],
    }
    result = pattern_beliefs(grid)
    expected = []
    for first, second in ((0, 0), (0, 1), (1, 0), (1, 1)):
        b = condition(prior_at_127, model.emission_matrix, np.array([first]))
        b = condition(
            b @ task.transition_matrix_for_action(1),
            model.emission_matrix, np.array([second]),
        )
        expected.append(b[0, 2])
    np.testing.assert_array_equal(result["step"], [128])
    np.testing.assert_allclose(result["rho"][0], expected)
    obs = observations(grid)
    np.testing.assert_array_equal(obs[0, 0, 3:], [0, 0, 0])
    np.testing.assert_array_equal(obs[0, 128, 3:], [0, 1, 0])


def test_episode_bootstrap_retains_sample_weighted_mean():
    result = bootstrap_mean(
        np.array([1.0, 1.0, 4.0]), np.array([0, 0, 1]), resamples=100,
    )
    assert result["mean"] == 2.0
    assert result["n_rows"] == 3
    assert result["n_episodes"] == 2


def test_global_AB_swap_preserves_exact_reward_state_evidence():
    model = control_model(alpha=0.85, transition_matrix=CYCLE_6_TRANSITION_MATRIX)
    task = ActionSymmetryTask(model=model, variant=2)
    original = model.initial_distribution[None, :]
    swapped = original.copy()
    tokens = [0, 0, 2, 1, 0, 1, 2]
    actions = [1, 0, 2, 1, 1, 0]
    for step, token in enumerate(tokens):
        alternate = 1 - token if token != 2 else 2
        original = condition(original, model.emission_matrix, np.array([token]))
        swapped = condition(swapped, model.emission_matrix, np.array([alternate]))
        np.testing.assert_allclose(original[:, 2], swapped[:, 2], atol=1e-14)
        if step < len(actions):
            transition = task.transition_matrix_for_action(actions[step])
            original = original @ transition
            swapped = swapped @ transition
