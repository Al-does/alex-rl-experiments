"""Optional next-token cross-entropy auxiliary objective for Pusher-B PPO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)
from learners.models.next_token import NextTokenAuxHead
from losses.next_token import NextTokenAuxLossMixin

NEXT_TOKEN_AUX_COEFFICIENT = 1.0


class ActorCriticWithNextTokenAux(
    NextTokenAuxHead,
    FactoredReproductionActorCritic,
):
    """Pusher-B actor-critic with a training-only next-token prediction head."""


class PPOWithNextTokenAux(NextTokenAuxLossMixin, PPOTorchLearner):
    """PPO plus coefficient-one next-token cross entropy."""


def next_token_targets(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align decision-time residuals with the pending token revealed one step later.

    The delayed observation at step t+1 contains the token the agent was asked
    to guess at step t, so aux logits at position t supervise on the token
    one-hot slice of the following observation row.
    """
    observations = batch[Columns.OBS]
    if observations.ndim != 3 or logits.ndim != 3:
        raise ValueError("next-token auxiliary training expects (B, T, D) tensors")
    num_classes = logits.shape[-1]
    if observations.shape[-1] < num_classes:
        raise ValueError("observations do not contain the token one-hot slice")
    next_tokens = observations[:, 1:, :num_classes]
    targets = next_tokens.argmax(dim=-1)
    populated = next_tokens.sum(dim=-1) > 0.5
    mask = batch.get(Columns.LOSS_MASK)
    if mask is None:
        mask = torch.ones(
            observations.shape[:2],
            dtype=torch.bool,
            device=observations.device,
        )
    else:
        mask = mask.to(device=observations.device, dtype=torch.bool)
    valid = mask[:, :-1] & mask[:, 1:] & populated
    return logits[:, :-1, :], targets, valid
