"""Experiment-local predictive Kelly head and auxiliary objective."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from ray.rllib.core.columns import Columns
from torch import nn


NAMESPACE = "reward_state_kelly"
TOKEN_LOGITS_KEY = f"{NAMESPACE}/token_logits"
WAGER_LOGITS_KEY = f"{NAMESPACE}/wager_logits"
TARGET_EXTRACTOR_KEY = f"{NAMESPACE}/target_extractor"
CORRECTNESS_COEFFICIENT_KEY = f"{NAMESPACE}/correctness_coefficient"
DIRECT_LOSS_COEFFICIENT_KEY = f"{NAMESPACE}/direct_loss_coefficient"
MAX_WAGER = 1.0 - 1e-4
NET_WIN_ODDS = 2.0

TargetExtractor = Callable[
    [Mapping[str, Any], torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class PredictiveKellyHead:
    """Add three token logits and three action-conditional wager logits."""

    def setup(self) -> None:
        super().setup()
        config = dict(self.model_config.get(NAMESPACE, {}))
        num_tokens = int(config.get("num_tokens", 3))
        if num_tokens <= 1:
            raise ValueError("predictive Kelly requires at least two tokens")
        self.reward_state_kelly_token_head = nn.Linear(
            self._embedding_dim,
            num_tokens,
        )
        self.reward_state_kelly_wager_head = nn.Linear(
            self._embedding_dim,
            num_tokens,
        )

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        embeddings = outputs[Columns.EMBEDDINGS]
        outputs[TOKEN_LOGITS_KEY] = self.reward_state_kelly_token_head(embeddings)
        outputs[WAGER_LOGITS_KEY] = self.reward_state_kelly_wager_head(embeddings)
        return outputs


def realized_log_growth(
    correct: torch.Tensor,
    wager: torch.Tensor,
) -> torch.Tensor:
    """Return fair three-way-bet log growth without leaving the device."""

    won = torch.log1p(wager * NET_WIN_ODDS)
    lost = torch.log1p(-wager)
    return torch.where(correct.to(dtype=torch.bool), won, lost)


def predictive_kelly_metrics(
    token_logits: torch.Tensor,
    wager_logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute correctness and selected-token Kelly losses on valid positions."""

    if token_logits.shape != wager_logits.shape:
        raise ValueError("token and wager logits must have matching shapes")
    if token_logits.shape[:-1] != targets.shape or targets.shape != valid.shape:
        raise ValueError("targets and validity must match logits leading axes")

    flat_logits = token_logits.reshape(-1, token_logits.shape[-1])
    flat_wager_logits = wager_logits.reshape(-1, wager_logits.shape[-1])
    flat_targets = targets.reshape(-1).to(
        device=flat_logits.device,
        dtype=torch.long,
    )
    weights = valid.reshape(-1).to(
        device=flat_logits.device,
        dtype=flat_logits.dtype,
    )
    count = weights.sum().clamp_min(1.0)

    selected_tokens = flat_logits.argmax(dim=-1)
    selected_wagers = torch.sigmoid(flat_wager_logits).gather(
        -1,
        selected_tokens.unsqueeze(-1),
    ).squeeze(-1).clamp(max=MAX_WAGER)
    correct = selected_tokens == flat_targets
    growth = realized_log_growth(correct, selected_wagers)
    cross_entropy = (
        F.cross_entropy(flat_logits, flat_targets, reduction="none") * weights
    ).sum() / count

    return {
        "cross_entropy": cross_entropy,
        "direct_loss": -(growth * weights).sum() / count,
        "accuracy": (correct.to(flat_logits.dtype) * weights).sum() / count,
        "selected_wager_mean": (selected_wagers * weights).sum() / count,
        "log_growth_mean": (growth * weights).sum() / count,
    }


class PredictiveKellyLossMixin:
    """Add token correctness and selected-token Kelly objectives to a Learner."""

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
        learner_config = config.learner_config_dict
        correctness_coefficient = float(
            learner_config.get(CORRECTNESS_COEFFICIENT_KEY, 0.0)
        )
        direct_coefficient = float(
            learner_config.get(DIRECT_LOSS_COEFFICIENT_KEY, 0.0)
        )
        if correctness_coefficient <= 0.0 or direct_coefficient <= 0.0:
            raise ValueError("Kelly arms require positive auxiliary coefficients")

        token_logits = fwd_out[TOKEN_LOGITS_KEY]
        wager_logits = fwd_out[WAGER_LOGITS_KEY]
        extractor: TargetExtractor | None = learner_config.get(
            TARGET_EXTRACTOR_KEY
        )
        if not callable(extractor):
            raise ValueError("Kelly arms require a callable token target extractor")
        aligned_logits, targets, valid = extractor(batch, token_logits)
        aligned_wagers = wager_logits[:, : aligned_logits.shape[1], :]
        metrics = predictive_kelly_metrics(
            aligned_logits,
            aligned_wagers,
            targets,
            valid,
        )
        self.metrics.log_dict(
            {
                f"{NAMESPACE}/cross_entropy": metrics["cross_entropy"],
                f"{NAMESPACE}/direct_loss": metrics["direct_loss"],
                f"{NAMESPACE}/accuracy": metrics["accuracy"],
                f"{NAMESPACE}/selected_wager_mean": metrics[
                    "selected_wager_mean"
                ],
                f"{NAMESPACE}/log_growth_mean": metrics["log_growth_mean"],
            },
            key=module_id,
            window=1,
        )
        return (
            total
            + correctness_coefficient * metrics["cross_entropy"]
            + direct_coefficient * metrics["direct_loss"]
        )
