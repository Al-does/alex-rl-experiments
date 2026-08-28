"""Flat product actions and factor-selective reward-state occupancy."""

from experiments.two_factor_reward_state_SAC_cycle_1.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
    CONDITIONS,
    N_ACTIONS,
    REWARD_STATE,
    TwoFactorShiftTask as _TwoFactorShiftTask,
    joint_transition,
    shifted_transition,
)


class TwoFactorShiftTask(_TwoFactorShiftTask):
    """The PR 65 controlled task, exposed from the PPO study namespace."""

