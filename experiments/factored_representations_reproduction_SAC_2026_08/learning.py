"""Next-joint-token auxiliary objective integrated with RLlib SAC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY

from losses.next_token import (
    FWD_KEY,
    LAMBDA_KEY,
    masked_classification_metrics,
)

AUXILIARY_COEFFICIENT = 1.0


def next_joint_token_targets(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the newly revealed token from SAC's next history observation."""

    next_observations = batch[Columns.NEXT_OBS]
    num_classes = logits.shape[-1]
    if next_observations.shape[-1] < num_classes:
        raise ValueError("next observations do not contain a joint-token slot")
    next_token = next_observations[..., :num_classes]
    targets = next_token.argmax(dim=-1)
    valid = next_token.sum(dim=-1) > 0.5
    return logits, targets, valid


class SACWithNextJointTokenAux(SACTorchLearner):
    """Add CE to SAC's actor loss while retaining split SAC optimizers.

    RLlib SAC backpropagates its named temporary actor/critic/temperature losses
    instead of the aggregate returned by ``compute_loss_for_module``. The
    auxiliary term must therefore be attached specifically to the actor loss.
    Its head lives inside ``pi_encoder``, so the standard actor optimizer owns
    all auxiliary parameters without touching either critic optimizer.
    """

    def compute_loss_for_module(
        self,
        *,
        module_id: str,
        config: Any,
        batch: dict[str, Any],
        fwd_out: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
        weight = float(config.learner_config_dict.get(LAMBDA_KEY, 0.0))
        if weight <= 0.0 or FWD_KEY not in fwd_out:
            return total

        logits, targets, valid = next_joint_token_targets(
            batch,
            fwd_out[FWD_KEY],
        )
        cross_entropy, accuracy = masked_classification_metrics(
            logits,
            targets,
            valid,
        )
        self.metrics.log_dict(
            {
                "next_token_aux/ce": cross_entropy,
                "next_token_aux/accuracy": accuracy,
            },
            key=module_id,
            window=1,
        )
        policy_key = (module_id, POLICY_LOSS_KEY)
        self._temp_losses[policy_key] = (
            self._temp_losses[policy_key] + weight * cross_entropy
        )
        return total + weight * cross_entropy
