"""Reference rewards for the two-factor reward-state control task."""

from __future__ import annotations

import numpy as np
from envs.hmm import HMMEnv
from envs.mess3.model import emission_matrix

from experiments.two_factor_reward_state_SAC_cycle_1.design import (
    AUDIT_BURN_IN,
    AUDIT_CHAINS,
    AUDIT_STEPS,
    GAMMA,
    demand_audit,
    fully_observed_occupancy,
    fully_observed_q_values,
)
from experiments.two_factor_reward_state_PPO_cycle_1.process import (
    MESS3_ALPHA,
    JOINT_STATE_COUNT,
    environment_config,
)
from experiments.two_factor_reward_state_PPO_cycle_1.task import (
    ACTION_PAIRS,
    REWARD_STATE,
    joint_transition,
)


def _joint_emission_matrix() -> np.ndarray:
    factor_emission = emission_matrix(MESS3_ALPHA)
    joint_emission = np.zeros((JOINT_STATE_COUNT, JOINT_STATE_COUNT))
    for state in range(JOINT_STATE_COUNT):
        factor_1 = state // 3
        factor_2 = state % 3
        for token in range(JOINT_STATE_COUNT):
            token_1 = token // 3
            token_2 = token % 3
            joint_emission[state, token] = (
                factor_emission[factor_1, token_1]
                * factor_emission[factor_2, token_2]
            )
    return joint_emission


def _reward_vector() -> np.ndarray:
    return (np.arange(JOINT_STATE_COUNT) // 3 == REWARD_STATE).astype(np.float64)


def _factored_optimal_joint_q_values() -> np.ndarray:
    """Q-values for reward_factor_1 under a factor-1-optimal joint policy."""

    transitions = np.stack([joint_transition(action) for action in range(9)])
    reward = _reward_vector()
    factor_values = fully_observed_q_values(gamma=GAMMA).max(axis=1)
    next_values = factor_values[np.arange(JOINT_STATE_COUNT) // 3]
    return reward[:, None] + GAMMA * np.einsum(
        "asj,j->as",
        transitions,
        next_values,
    )


def _shift_to_flat_action(*, factor_2_shift: int = 0) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for action, (shift_1, shift_2) in enumerate(ACTION_PAIRS):
        if shift_2 == factor_2_shift and shift_1 not in mapping:
            mapping[shift_1] = action
    return mapping


def fully_observed_reward_factor_1(*, steps: int = 200_000, seed: int = 20260828) -> float:
    """Mean per-step reward under factor-1-optimal control with visible states."""

    policy_1 = fully_observed_q_values(gamma=GAMMA).argmax(axis=1)
    action_for_shift = _shift_to_flat_action(factor_2_shift=0)
    config = environment_config("reward_factor_1")
    config["diagnostics"] = {"state": True}
    env = HMMEnv(config)
    rewards: list[float] = []
    _, info = env.reset(seed=seed)
    for _ in range(steps):
        factor_1 = info["state_current"] // 3
        _, reward, _, truncated, info = env.step(action_for_shift[policy_1[factor_1]])
        rewards.append(reward)
        if truncated:
            _, info = env.reset()
    return float(np.mean(rewards))


def two_factor_qmdp_reward_factor_1(
    *,
    steps: int = 200_000,
    seed: int = 20260828,
) -> float:
    """Mean per-step reward under exact-filter QMDP control in the two-factor env."""

    q_values = _factored_optimal_joint_q_values()
    config = environment_config("reward_factor_1")
    config["diagnostics"] = {"belief": True}
    env = HMMEnv(config)
    rewards: list[float] = []
    _, info = env.reset(seed=seed)
    for _ in range(steps):
        belief = info["belief_current"]
        _, reward, _, truncated, info = env.step(int((belief @ q_values).argmax()))
        rewards.append(reward)
        if truncated:
            _, info = env.reset()
    return float(np.mean(rewards))


def bayes_max_reward_factor_1(*, seed: int = 20260828) -> float:
    """Bayes-optimal mean per-step reward for reward_factor_1.

    The preregistered partial-observability audit simulates the single-factor
    QMDP benchmark that this study family uses as the Bayes reference. The
    fully observed factor-1 ceiling in the actual two-factor environment is
    higher (~5/9), but that requires latent-state access rather than filtering
    joint tokens.
    """

    del seed
    return float(demand_audit()["qmdp"])


def bayes_max_reward_occupancy_factor_1() -> float:
    """Fully observed oracle factor-1 reward-state-2 occupancy."""

    return float(fully_observed_occupancy())


def reference_rewards(*, seed: int = 20260828) -> dict[str, float]:
    """Return the main reference reward ceilings used in plots and write-ups."""

    return {
        "bayes_max_reward": bayes_max_reward_factor_1(seed=seed),
        "audit_qmdp_reward": float(demand_audit()["qmdp"]),
        "two_factor_qmdp_reward": two_factor_qmdp_reward_factor_1(seed=seed),
        "fully_observed_reward": fully_observed_reward_factor_1(seed=seed),
    }


__all__ = [
    "AUDIT_BURN_IN",
    "AUDIT_CHAINS",
    "AUDIT_STEPS",
    "bayes_max_reward_factor_1",
    "bayes_max_reward_occupancy_factor_1",
    "fully_observed_reward_factor_1",
    "reference_rewards",
    "two_factor_qmdp_reward_factor_1",
]
