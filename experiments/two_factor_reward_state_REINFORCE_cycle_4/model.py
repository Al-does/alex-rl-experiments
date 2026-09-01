"""Two-factor transformer policy with an identically zero value baseline."""

from typing import Any

import torch
from ray.rllib.core.columns import Columns

from experiments.two_factor_reward_state_PPO_cycle_2.model import (
    TwoFactorRewardPPO,
)


class TwoFactorRewardReinforceCycle4(TwoFactorRewardPPO):
    """Cycle-3 architecture used as a baseline-free REINFORCE policy."""

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference = embeddings if embeddings is not None else batch[Columns.OBS]
        return torch.zeros(
            reference.shape[:-1],
            dtype=reference.dtype,
            device=reference.device,
        )
