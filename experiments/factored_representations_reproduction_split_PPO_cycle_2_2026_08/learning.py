"""Auxiliary-learning pieces shared with the matched PPO reproduction."""

from experiments.factored_representations_reproduction_PPO_2026_08.learning import (
    AUXILIARY_COEFFICIENT,
    PPOWithNextJointTokenAux,
    next_joint_token_targets,
)

__all__ = [
    "AUXILIARY_COEFFICIENT",
    "PPOWithNextJointTokenAux",
    "next_joint_token_targets",
]

