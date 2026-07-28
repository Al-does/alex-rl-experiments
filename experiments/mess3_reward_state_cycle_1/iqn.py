"""Experiment-local IQN value head and PPO loss composition."""

# Deprecated for new experiments. Preserve this module so historical recipes
# remain reproducible; new work should compose `learners.models.IQNValueMixin`
# with a base model, select `learners.IQNPPOTorchLearner`, and use
# `losses.quantile_huber_loss` for standalone objective math.

from __future__ import annotations

from typing import Any

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from torch import nn

from learners.models.transformer import TransformerModel


NAMESPACE = "iqn_value"
FWD_QUANTILES = f"{NAMESPACE}/quantiles"
FWD_TAUS = f"{NAMESPACE}/taus"
LOSS_COEFFICIENT_KEY = f"{NAMESPACE}/loss_coefficient"
HUBER_KAPPA_KEY = f"{NAMESPACE}/huber_kappa"


class IQNValueHead(nn.Module):
    """Implicit quantile head conditioned on transformer embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        n_cosines: int,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or n_cosines <= 0:
            raise ValueError("IQN dimensions must be positive")
        self.embedding_dim = int(embedding_dim)
        self.n_cosines = int(n_cosines)
        self.cosine_projection = nn.Linear(n_cosines, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        self.register_buffer(
            "cosine_frequencies",
            torch.arange(1, n_cosines + 1, dtype=torch.float32) * torch.pi,
            persistent=False,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.shape[:-1] != taus.shape[:-1]:
            raise ValueError("embedding and tau leading dimensions must match")
        frequencies = self.cosine_frequencies.to(dtype=embeddings.dtype)
        cosine_features = torch.cos(taus.unsqueeze(-1) * frequencies)
        tau_embeddings = torch.relu(
            self.cosine_projection(cosine_features)
        )
        joint = embeddings.unsqueeze(-2) * tau_embeddings
        return self.output(joint).squeeze(-1)


class IQNTransformerModel(TransformerModel):
    """Continuous policy with an implicit-quantile value distribution."""

    def setup(self) -> None:
        super().setup()
        config = dict(self.model_config.get(NAMESPACE, {}))
        self.train_quantiles = int(config.get("train_quantiles", 32))
        self.value_quantiles_count = int(config.get("value_quantiles", 64))
        n_cosines = int(config.get("n_cosines", 64))
        if self.train_quantiles <= 0 or self.value_quantiles_count <= 0:
            raise ValueError("IQN quantile counts must be positive")
        self.iqn_value_head = IQNValueHead(
            self._embedding_dim,
            n_cosines=n_cosines,
        )
        self.heads.value = nn.Identity()

    def _sample_taus(self, embeddings: torch.Tensor, count: int) -> torch.Tensor:
        return torch.rand(
            (*embeddings.shape[:-1], count),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )

    def _fixed_taus(self, embeddings: torch.Tensor, count: int) -> torch.Tensor:
        midpoints = (
            torch.arange(
                count,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            + 0.5
        ) / count
        return midpoints.expand(*embeddings.shape[:-1], count)

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        embeddings = outputs[Columns.EMBEDDINGS]
        taus = self._sample_taus(embeddings, self.train_quantiles)
        outputs[FWD_TAUS] = taus
        outputs[FWD_QUANTILES] = self.iqn_value_head(embeddings, taus)
        return outputs

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings, _ = self._encode_train(batch)
        taus = self._fixed_taus(
            embeddings,
            self.value_quantiles_count,
        )
        return self.iqn_value_head(embeddings, taus).mean(dim=-1)


def quantile_huber_loss(
    quantiles: torch.Tensor,
    taus: torch.Tensor,
    targets: torch.Tensor,
    *,
    kappa: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile-Huber regression against sampled on-policy return targets."""

    if kappa <= 0.0:
        raise ValueError("quantile Huber kappa must be positive")
    if quantiles.shape != taus.shape:
        raise ValueError("quantiles and taus must have matching shapes")
    if quantiles.shape[:-1] != targets.shape:
        raise ValueError("targets must match quantile leading dimensions")
    errors = targets.unsqueeze(-1) - quantiles
    absolute_errors = errors.abs()
    huber = torch.where(
        absolute_errors <= kappa,
        0.5 * errors.square(),
        kappa * (absolute_errors - 0.5 * kappa),
    )
    weights = (
        taus - (errors.detach() < 0.0).to(dtype=taus.dtype)
    ).abs()
    per_item = (weights * huber / kappa).mean(dim=-1)
    if valid is None:
        return per_item.mean()
    valid_float = valid.to(device=per_item.device, dtype=per_item.dtype)
    return (per_item * valid_float).sum() / valid_float.sum().clamp_min(1.0)


class IQNPPOTorchLearner(PPOTorchLearner):
    """PPO with policy loss plus an IQN distributional value objective."""

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
        coefficient = float(
            learner_config.get(LOSS_COEFFICIENT_KEY, 0.5)
        )
        kappa = float(learner_config.get(HUBER_KAPPA_KEY, 1.0))
        if coefficient <= 0.0:
            raise ValueError("IQN loss coefficient must be positive")

        quantiles = fwd_out[FWD_QUANTILES]
        taus = fwd_out[FWD_TAUS]
        targets = batch[Postprocessing.VALUE_TARGETS]
        valid = batch.get(Columns.LOSS_MASK)
        iqn_loss = quantile_huber_loss(
            quantiles,
            taus,
            targets,
            kappa=kappa,
            valid=valid,
        )
        spread = quantiles.std(dim=-1, correction=0)
        if valid is None:
            mean_spread = spread.mean()
        else:
            valid_float = valid.to(
                device=spread.device,
                dtype=spread.dtype,
            )
            mean_spread = (
                (spread * valid_float).sum()
                / valid_float.sum().clamp_min(1.0)
            )
        self.metrics.log_dict(
            {
                f"{NAMESPACE}/loss": iqn_loss,
                f"{NAMESPACE}/mean_quantile_spread": mean_spread,
            },
            key=module_id,
            window=1,
        )
        return total + coefficient * iqn_loss
