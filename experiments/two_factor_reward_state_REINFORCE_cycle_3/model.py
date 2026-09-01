"""Cycle-2 RoPE transformer policy and value baseline for REINFORCE."""

from experiments.two_factor_reward_state_PPO_cycle_2.model import (
    TwoFactorRewardPPO,
)


class TwoFactorRewardReinforce(TwoFactorRewardPPO):
    """Cycle-2 architecture trained with the cycle-3 REINFORCE objective."""
