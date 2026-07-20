"""PPO extension that puts behavior-policy entropy into sampled rewards."""

from __future__ import annotations

from typing import Any

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from ray.rllib.core.columns import Columns


COEFFICIENT_KEY = "entropy_reward/coefficient"


def add_categorical_entropy_reward(
    rewards: torch.Tensor,
    logits: torch.Tensor,
    *,
    coefficient: float,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add detached categorical entropy bonuses without leaving the device."""

    if coefficient < 0.0:
        raise ValueError("entropy reward coefficient must be non-negative")
    entropy = torch.distributions.Categorical(logits=logits).entropy().detach()
    if entropy.shape != rewards.shape:
        raise ValueError("policy entropy and rewards must have matching shapes")
    bonus = entropy * coefficient
    if valid is not None:
        if valid.shape != rewards.shape:
            raise ValueError("valid mask and rewards must have matching shapes")
        bonus = bonus * valid.to(device=bonus.device, dtype=bonus.dtype)
    return rewards + bonus, bonus


class AddPolicyEntropyToRewards(ConnectorV2):
    """Augment rewards before PPO computes GAE and value targets."""

    def __init__(self, *, coefficient: float) -> None:
        super().__init__()
        if coefficient <= 0.0:
            raise ValueError("entropy reward coefficient must be positive")
        self.coefficient = float(coefficient)

    @torch.no_grad()
    def __call__(
        self,
        *,
        rl_module,
        episodes,
        batch: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        del rl_module, episodes, kwargs
        for module_batch in batch.values():
            rewards = module_batch[Columns.REWARDS]
            logits = module_batch[Columns.ACTION_DIST_INPUTS]
            valid = module_batch.get(Columns.LOSS_MASK)
            augmented, _ = add_categorical_entropy_reward(
                rewards,
                logits,
                coefficient=self.coefficient,
                valid=valid,
            )
            module_batch[Columns.REWARDS] = augmented
        return batch


class EntropyRewardPPOTorchLearner(PPOTorchLearner):
    """PPO Learner whose critic and GAE use entropy-augmented rewards."""

    def build(self) -> None:
        super().build()
        coefficient = float(
            self.config.learner_config_dict.get(COEFFICIENT_KEY, 0.0)
        )
        if coefficient <= 0.0:
            raise ValueError(
                f"{COEFFICIENT_KEY!r} must be positive for max-entropy PPO"
            )
        self._learner_connector.insert_before(
            GeneralAdvantageEstimation,
            AddPolicyEntropyToRewards(coefficient=coefficient),
        )
