"""Device-native differential average-reward extension for IQN PPO."""

from __future__ import annotations

from typing import Any

import torch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from ray.rllib.core.columns import Columns

from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    FWD_QUANTILES,
    IQNPPOTorchLearner,
)


NAMESPACE = "average_reward"
EMA_DECAY_KEY = f"{NAMESPACE}/ema_decay"
VALUE_ANCHOR_COEFFICIENT_KEY = f"{NAMESPACE}/value_anchor_coefficient"


def center_rewards(
    rewards: torch.Tensor,
    average_reward: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Subtract the reward rate while preserving padded artificial timesteps."""

    if average_reward.numel() != 1:
        raise ValueError("average reward estimate must be scalar")
    if valid is None:
        return rewards - average_reward
    if valid.shape != rewards.shape:
        raise ValueError("valid mask and rewards must have matching shapes")
    valid_float = valid.to(device=rewards.device, dtype=rewards.dtype)
    return rewards - average_reward * valid_float


class CenterRewardsByRunningAverage(ConnectorV2):
    """Estimate and subtract each module's long-run reward rate before GAE."""

    def __init__(self, *, ema_decay: float) -> None:
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("average-reward EMA decay must lie in [0, 1)")
        self.ema_decay = float(ema_decay)
        self.estimates: dict[str, torch.Tensor] = {}

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
        for module_id, module_batch in batch.items():
            rewards = module_batch[Columns.REWARDS]
            valid = module_batch.get(Columns.LOSS_MASK)
            if valid is None:
                batch_average = rewards.mean()
            else:
                valid_float = valid.to(
                    device=rewards.device,
                    dtype=rewards.dtype,
                )
                batch_average = (
                    (rewards * valid_float).sum()
                    / valid_float.sum().clamp_min(1.0)
                )
            batch_average = batch_average.detach()
            previous = self.estimates.get(module_id)
            estimate = (
                batch_average
                if previous is None
                else (
                    self.ema_decay * previous
                    + (1.0 - self.ema_decay) * batch_average
                )
            )
            self.estimates[module_id] = estimate
            module_batch[Columns.REWARDS] = center_rewards(
                rewards,
                estimate,
                valid=valid,
            )
        return batch


class AverageRewardIQNPPOTorchLearner(IQNPPOTorchLearner):
    """IQN PPO with differential rewards and a value-location anchor."""

    def build(self) -> None:
        super().build()
        learner_config = self.config.learner_config_dict
        ema_decay = float(learner_config.get(EMA_DECAY_KEY, 0.95))
        self.average_reward_connector = CenterRewardsByRunningAverage(
            ema_decay=ema_decay
        )
        self._learner_connector.insert_before(
            GeneralAdvantageEstimation,
            self.average_reward_connector,
        )

    def compute_loss_for_module(
        self,
        *,
        module_id,
        config,
        batch,
        fwd_out,
    ):
        total = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
        coefficient = float(
            config.learner_config_dict.get(
                VALUE_ANCHOR_COEFFICIENT_KEY,
                0.01,
            )
        )
        if coefficient < 0.0:
            raise ValueError("value anchor coefficient must be non-negative")
        scalar_values = fwd_out[FWD_QUANTILES].mean(dim=-1)
        valid = batch.get(Columns.LOSS_MASK)
        if valid is None:
            mean_value = scalar_values.mean()
        else:
            valid_float = valid.to(
                device=scalar_values.device,
                dtype=scalar_values.dtype,
            )
            mean_value = (
                (scalar_values * valid_float).sum()
                / valid_float.sum().clamp_min(1.0)
            )
        anchor_loss = mean_value.square()
        average_reward = self.average_reward_connector.estimates.get(module_id)
        metrics = {f"{NAMESPACE}/value_anchor_loss": anchor_loss}
        if average_reward is not None:
            metrics[f"{NAMESPACE}/estimate"] = average_reward
        self.metrics.log_dict(
            metrics,
            key=module_id,
            window=1,
        )
        return total + coefficient * anchor_loss
