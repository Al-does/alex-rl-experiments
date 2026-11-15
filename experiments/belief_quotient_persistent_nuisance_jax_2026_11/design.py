"""Exact stationary model-based references (not Bayes-optimality bounds)."""
import itertools
import json

import numpy as np

from experiments.belief_quotient_persistent_nuisance_jax_2026_11.shared import (
    action_transitions, emission_matrix,
)


def stationary_distribution(transition):
    system = transition.T - np.eye(len(transition))
    system[-1] = 1
    target = np.zeros(len(transition))
    target[-1] = 1
    return np.linalg.solve(system, target)


def history_belief(tokens, kernels=None):
    kernels = action_transitions(2) if kernels is None else kernels
    emission = emission_matrix()
    belief = np.full(3, 1 / 3)
    for index, token in enumerate(tokens):
        if index:
            belief = belief @ kernels[1]
        belief *= emission[:, token]
        belief /= belief.sum()
    return belief


def controller_references():
    emission, kernels = emission_matrix(), action_transitions(2)
    policies = list(itertools.product(range(3), repeat=3))
    rewards = []
    for policy in policies:
        transition = sum(emission[:, token, None] * kernels[action]
                         for token, action in enumerate(policy))
        rewards.append(float(stationary_distribution(transition)[2]))
    transition = np.zeros((27, 27))
    for state, older, previous, token, next_state in itertools.product(range(3), repeat=5):
        action = 0 if older == previous == token == 2 else 1
        transition[9 * state + 3 * older + previous,
                   9 * next_state + 3 * previous + token] += (
                       emission[state, token] * kernels[action, state, next_state])
    history = float(stationary_distribution(transition)[18:].sum())
    u, v = kernels[:, 0, 2], kernels[:, 2, 2]
    U = .25 * u[1] + .75 * u
    fractions = U / (U + 1 - v)
    best = int(np.argmax(rewards))
    return {"current_token_policy": policies[best], "current_token": rewards[best],
            "c_action_fractional_rewards": fractions.tolist(),
            "feasible_three_token": history, "relative_gain": history / rewards[best] - 1,
            "AAAC": history_belief([0, 0, 0, 2]).tolist(),
            "BBAC": history_belief([1, 1, 0, 2]).tolist()}


if __name__ == "__main__":
    print(json.dumps(controller_references(), indent=2))
