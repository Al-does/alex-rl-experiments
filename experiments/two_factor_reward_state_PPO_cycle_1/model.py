"""Stateful 64-dimensional transformer actor-critic for two-factor PPO."""

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
    FactoredReproductionModelConfig,
    ReproductionResidualEncoder,
)


class TwoFactorRewardPPO(FactoredReproductionActorCritic):
    """PPO actor-critic over aligned joint-token/preceding-action frames."""


__all__ = [
    "FactoredReproductionModelConfig",
    "ReproductionResidualEncoder",
    "TwoFactorRewardPPO",
]
