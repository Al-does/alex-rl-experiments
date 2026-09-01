"""PPO namespace for the shared two-factor variant-3 task."""

from experiments.two_factor_reward_state_SAC_cycle_2.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
    CONDITIONS,
    N_ACTIONS,
    REWARD_STATE,
    VARIANT_3_DIRECTIONS,
    TwoFactorVariant3Task,
    factor_transition,
    joint_transition,
)


__all__ = [
    "ACTION_LABELS",
    "ACTION_PAIRS",
    "CONDITIONS",
    "N_ACTIONS",
    "REWARD_STATE",
    "VARIANT_3_DIRECTIONS",
    "TwoFactorVariant3Task",
    "factor_transition",
    "joint_transition",
]
