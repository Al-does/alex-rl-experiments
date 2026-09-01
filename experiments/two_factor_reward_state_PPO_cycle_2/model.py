"""Cycle-5 transformer actor-critic for the two-factor PPO study."""

from learners.models.transformer import TransformerModel


class TwoFactorRewardPPO(TransformerModel):
    """RoPE actor-critic with dimensions inferred from joint observations/actions."""
