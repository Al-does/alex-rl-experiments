"""Optional next-joint-token auxiliary objective for the PPO comparison arm."""

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


AUXILIARY_COEFFICIENT = 1.0


class ActorCriticWithNextJointTokenAux(
    NextTokenAuxHead,
    FactoredReproductionActorCritic,
):
    """PPO actor-critic with a training-only joint-token prediction head."""


class PPOWithNextJointTokenAux(NextTokenAuxLossMixin, PPOTorchLearner):
    """Standard PPO plus coefficient-one next-joint-token cross entropy."""


def next_joint_token_targets(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align decision-time residuals with the token revealed one step later."""

    observations = batch[Columns.OBS]
    if observations.ndim != 3 or logits.ndim != 3:
        raise ValueError("next-token auxiliary training expects (B, T, D) tensors")
    num_classes = logits.shape[-1]
    if observations.shape[-1] < num_classes:
        raise ValueError("observations do not contain the joint-token one-hot slice")
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
