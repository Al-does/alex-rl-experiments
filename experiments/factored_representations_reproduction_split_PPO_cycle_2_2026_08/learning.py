"""Auxiliary-learning pieces shared with the matched PPO reproduction."""

from experiments.factored_representations_reproduction_PPO_2026_08.learning import (
    AUXILIARY_COEFFICIENT,
    PPOWithNextJointTokenAux,
    next_joint_token_targets,
)
from experiments.mess3_token_guess_cycle_1.entropy_reward import (
    COEFFICIENT_KEY as ENTROPY_REWARD_COEFFICIENT_KEY,
    EntropyRewardPPOTorchLearner,
)

__all__ = [
    "AUXILIARY_COEFFICIENT",
    "ENTROPY_REWARD_COEFFICIENT_KEY",
    "EntropyRewardPPOTorchLearner",
    "PPOWithNextJointTokenAux",
    "next_joint_token_targets",
]

