"""Stationary current-token optima and constructive history references."""

import itertools
import json

import numpy as np

from experiments.belief_quotient_history_jax_lumped_2026_11.shared import (
    action_transitions,
    emission_matrix,
)


def stationary_distribution(transition):
    system = transition.T - np.eye(len(transition))
    system[-1] = 1
    target = np.zeros(len(transition))
    target[-1] = 1
    return np.linalg.solve(system, target)


def controller_references():
    emission = emission_matrix()
    results = {}
    for variant in (2, 3):
        kernels = action_transitions(variant)
        reactive = []
        for policy in itertools.product(range(3), repeat=3):
            transition = sum(emission[:, token, None] * kernels[action]
                             for token, action in enumerate(policy))
            reactive.append(float(stationary_distribution(transition)[2]))
        transition = np.zeros((27, 27))
        for state, older, previous, token, next_state in itertools.product(range(3), repeat=5):
            if variant == 2:
                action = 0 if token == 2 or older == previous == 2 else 1
            else:
                action = 0 if previous == 2 else (1, 2, 0)[token]
            transition[9 * state + 3 * older + previous,
                       9 * next_state + 3 * previous + token] = (
                emission[state, token] * kernels[action, state, next_state])
        history = float(stationary_distribution(transition)[18:].sum())
        results[variant] = {"current_token": max(reactive), "feasible_three_token": history,
                            "relative_gain": history / max(reactive) - 1}
    return results


if __name__ == "__main__":
    print(json.dumps(controller_references(), indent=2))
