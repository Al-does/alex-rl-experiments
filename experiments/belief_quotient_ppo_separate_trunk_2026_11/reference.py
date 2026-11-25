"""Finite-horizon discounted belief-grid probe targets for the eps02 ladder."""
import numpy as np

from envs.mess3.solvers.simplex_grid import interp_weights, simplex_grid


REWARD_REFERENCES = {
    1: {"full_episode_optimal": 0.941335636, "full_episode_bayes_upper": 0.941335636},
    3: {"full_episode_feasible": 0.420969, "full_episode_bayes_upper": 0.421412617},
}


def value_grid(process, gamma, horizon, divisions):
    grid = simplex_grid(divisions)
    predictive = np.einsum("gs,ast->gat", grid, np.asarray(process.transitions, dtype=float))
    joint = predictive[..., None, :] * np.asarray(process.emission, dtype=float).T
    probabilities = joint.sum(-1)
    posterior = joint / np.maximum(probabilities[..., None], 1e-300)
    indices, weights = interp_weights(posterior, divisions)
    coefficients = weights * probabilities[..., None]
    rewards = grid @ np.asarray(process.reward, dtype=float)
    values = np.zeros((horizon + 1, len(grid)))
    for remaining in range(1, horizon + 1):
        continuation = (values[remaining - 1][indices] * coefficients).sum(axis=(-1, -2))
        values[remaining] = rewards + gamma * continuation.max(-1)
    return values


def value_target(values, beliefs, divisions):
    horizon = values.shape[0] - 1
    remaining = np.arange(horizon, 0, -1)
    indices, weights = interp_weights(beliefs, divisions)
    return (values[remaining[:, None], indices] * weights).sum(-1)[..., None]
