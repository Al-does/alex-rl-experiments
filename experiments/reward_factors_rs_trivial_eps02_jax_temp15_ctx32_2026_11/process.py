"""Reward-State x Trivial independent factors (eps02 persistent-nuisance factors).

Factor 1 uses the variant-2 (Reward-State) tilts: actions 1/2 tilt the odds of
entering state 2 from states 0/1 up/down, and both tilt state 2 towards leaving.
Factor 2 uses the variant-1 (Trivial) tilts: action 1 raises state-2 odds from
every state. Both factors share the eps02 T0/E, are observed through their own
token (joint token = 3 * token_1 + token_2) and are rewarded additively by
pre-transition state-2 occupancy. Joint actions use the source ACTION_PAIRS order.
"""
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.reward_factors_history_jax_2026_11.process import ACTION_PAIRS

REWARD_STATE_VARIANT = 2
TRIVIAL_VARIANT = 1
REWARD_STATE = environment.REWARD_STATE


def factor_matrices():
    return (environment.action_transitions(REWARD_STATE_VARIANT),
            environment.action_transitions(TRIVIAL_VARIANT), environment.emission_matrix())


def rs_trivial_process():
    first_factor, second_factor, emission = factor_matrices()
    transitions = np.stack([np.kron(first_factor[a], second_factor[b]) for a, b in ACTION_PAIRS])
    first, second = np.divmod(np.arange(9), 3)
    components = np.stack([first == REWARD_STATE, second == REWARD_STATE]).astype(float)
    return make_process(np.full(9, 1 / 9), transitions, np.kron(emission, emission),
                        components.sum(0), components)


def factor_marginals(belief):
    joint = np.asarray(belief).reshape(-1, 3, 3)
    return joint.sum(2), joint.sum(1)


def make_targets(value_fn):
    """Probe targets; ``value_fn(factor_1, factor_2)`` is the Bayes-optimal (1-gamma)V*."""

    def targets(belief):
        first, second = factor_marginals(belief)
        return {
            "joint": np.asarray(belief), "factor_1": first, "factor_2": second,
            "rho_1": first[:, 2:], "rho_2": second[:, 2:],
            "delta_1": first[:, :1] - first[:, 1:2], "delta_2": second[:, :1] - second[:, 1:2],
            "value_star": value_fn(first, second)[:, None],
        }
    return targets


def environment_metadata():
    first_factor, second_factor, emission = factor_matrices()
    return {
        **environment.environment_metadata(),
        "factor_1": {"name": "Reward-State", "variant": REWARD_STATE_VARIANT,
                     "transitions": first_factor.tolist()},
        "factor_2": {"name": "Trivial", "variant": TRIVIAL_VARIANT,
                     "transitions": second_factor.tolist()},
        "factor_emission_matrix": emission.tolist(),
        "action_pairs": ACTION_PAIRS,
        "token_order": "3 * token_1 + token_2",
        "reward": "pre-transition state-2 occupancy of factor 1 plus factor 2",
    }
