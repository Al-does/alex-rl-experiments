"""Two-factor transformer policy with an identically zero value baseline."""

from typing import Any

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

from experiments.two_factor_reward_state_PPO_cycle_2.model import (
    TwoFactorRewardPPO,
)


class TwoFactorRewardReinforceCycle4(TwoFactorRewardPPO):
    """Cycle-3 architecture used as a baseline-free REINFORCE policy."""

    @override(TorchRLModule)
    def setup(self):
        super().setup()
        self._sampling_temperature = float(
            self.model_config.get("sampling_temperature", 1.0)
        )
        if self._sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")

    def _outputs(
        self,
        embeddings: torch.Tensor,
        state_out: Any | None,
        *,
        training: bool,
    ) -> dict[str, Any]:
        outputs = super()._outputs(embeddings, state_out, training=training)
        temperature = self._sampling_temperature
        if temperature != 1.0:
            outputs[Columns.ACTION_DIST_INPUTS] = (
                outputs[Columns.ACTION_DIST_INPUTS] / temperature
            )
        return outputs

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
