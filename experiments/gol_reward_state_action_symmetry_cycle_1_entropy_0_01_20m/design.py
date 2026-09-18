from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from envs.gol.model import ACTIONS, AGGREGATION, STATES, controlled_kernels, gol_model
from envs.hmm.model import stationary_distribution
from experiments.gol_reward_state_action_symmetry_cycle_1_entropy_0_01_20m.process import SPEED, VARIANTS


def analytic_design_summary(speed: str = SPEED) -> dict[str, Any]:
    variants = {}
    for variant in VARIANTS:
        kernel = controlled_kernels(variant=variant, speed=speed)
        transitions = kernel.sum(axis=1)
        rewards = transitions[:, :, 2].T
        emissions = kernel.sum(axis=-1).transpose(0, 2, 1)
        null = np.array([0.0, 0.0, 1.0, -1.0])
        ranked = []
        for policy in itertools.product(range(4), repeat=4):
            matrix = np.stack([transitions[action, state] for state, action in enumerate(policy)])
            ranked.append((float(stationary_distribution(matrix)[2]), policy))
        occupancy, policy = max(ranked, key=lambda item: item[0])
        expected = (0, 0 if variant == 2 else 1, 2, 3)
        if policy != expected:
            raise AssertionError(f"unexpected full-information policy: {policy}")
        variants[f"variant_{variant}"] = {
            "reset_prior": gol_model(variant=variant, speed=speed).initial_distribution.tolist(),
            "full_information_policy": list(policy),
            "full_information_occupancy_upper_bound": occupancy,
            "uniform_random_action_occupancy": float(stationary_distribution(transitions.mean(axis=0))[2]),
            "constant_action_occupancies": [float(stationary_distribution(matrix)[2]) for matrix in transitions],
            "reward_prediction_matrix": rewards.tolist(),
            "reward_prediction_rank": int(np.linalg.matrix_rank(rewards)),
            "marginal_null_error": float(max(np.abs(null @ rewards).max(), np.abs(null @ emissions).max())),
        }
    fine = controlled_kernels(variant=2, speed=speed)
    coarse = controlled_kernels(variant=2, speed=speed, coarse=True)
    broken = controlled_kernels(variant=3, speed=speed) @ AGGREGATION
    return {
        "spec_version": "1.0",
        "speed": speed,
        "states": list(STATES),
        "actions": list(ACTIONS),
        "variants": variants,
        "variant_2_exact_quotient_error": float(np.abs(fine @ AGGREGATION - AGGREGATION @ coarse).max()),
        "variant_3_aggregation_error": float(np.abs(broken[:, :, 0] - broken[:, :, 1]).max()),
        "marginal_null_direction": [0, 0, 1, -1],
        "benchmark_scope": "Full-information long-run bounds, not certified Bayes-optimal POMDP rewards.",
    }
