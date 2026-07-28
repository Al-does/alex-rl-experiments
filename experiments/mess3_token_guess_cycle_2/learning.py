"""A2C and decoupled Kelly objectives for the token-guess battery."""

from __future__ import annotations

from typing import Any

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import (
    ENTROPY_KEY,
    POLICY_LOSS_KEY,
    VF_LOSS_KEY,
)
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.torch_utils import explained_variance
from torch import nn


KELLY_NAMESPACE = "token_guess_kelly"
KELLY_LOGITS_KEY = f"{KELLY_NAMESPACE}/logits"
KELLY_CORRECT_KEY = f"{KELLY_NAMESPACE}/correct"
KELLY_LOSS_COEFFICIENT_KEY = f"{KELLY_NAMESPACE}/loss_coefficient"
MAX_WAGER = 1.0 - 1e-4
NET_WIN_ODDS = 2.0


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def a2c_objective(
    *,
    logp: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    value_targets: torch.Tensor,
    entropy: torch.Tensor,
    loss_mask: torch.Tensor | None,
    vf_loss_coeff: float,
    entropy_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the synchronous one-pass advantage actor-critic objective."""

    policy_loss = -_masked_mean(logp * advantages.detach(), loss_mask)
    value_loss = _masked_mean(
        0.5 * (values - value_targets.detach()).square(),
        loss_mask,
    )
    mean_entropy = _masked_mean(entropy, loss_mask)
    total = (
        policy_loss
        + vf_loss_coeff * value_loss
        - entropy_coeff * mean_entropy
    )
    return total, policy_loss, value_loss, mean_entropy


class A2CTorchLearner(PPOTorchLearner):
    """A2C loss on PPO's synchronous sampling and normalized-GAE pipeline."""

    def build(self) -> None:
        config = self.config
        if not config.enable_rl_module_and_learner:
            raise ValueError("A2C requires RLModule/Learner")
        if not config.enable_env_runner_and_connector_v2:
            raise ValueError("A2C requires EnvRunner/ConnectorV2")
        if not config.add_default_connectors_to_learner_pipeline:
            raise ValueError("A2C requires PPO's GAE connector")
        if not config.use_critic or not config.use_gae:
            raise ValueError("A2C requires GAE with a critic")
        if config.use_kl_loss:
            raise ValueError("A2C does not use a KL penalty")
        if config.num_epochs != 1 or config.minibatch_size is not None:
            raise ValueError("A2C requires one full-batch optimizer update")
        if config.shuffle_batch_per_epoch:
            raise ValueError("A2C does not shuffle or reuse its fresh batch")
        super().build()

    def compute_loss_for_module(
        self,
        *,
        module_id,
        config,
        batch,
        fwd_out,
    ):
        module = self.module[module_id].unwrapped()
        action_dist = module.get_train_action_dist_cls().from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        values = module.compute_values(
            batch,
            embeddings=fwd_out.get(Columns.EMBEDDINGS),
        )
        entropy_coeff = self.entropy_coeff_schedulers_per_module[
            module_id
        ].get_current_value()
        total, policy_loss, value_loss, mean_entropy = a2c_objective(
            logp=action_dist.logp(batch[Columns.ACTIONS]),
            advantages=batch[Postprocessing.ADVANTAGES],
            values=values,
            value_targets=batch[Postprocessing.VALUE_TARGETS],
            entropy=action_dist.entropy(),
            loss_mask=batch.get(Columns.LOSS_MASK),
            vf_loss_coeff=float(config.vf_loss_coeff),
            entropy_coeff=float(entropy_coeff),
        )
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: policy_loss,
                VF_LOSS_KEY: value_loss,
                ENTROPY_KEY: mean_entropy,
                "a2c/vf_explained_var": explained_variance(
                    batch[Postprocessing.VALUE_TARGETS],
                    values,
                ),
            },
            key=module_id,
            window=1,
        )
        return total


class DecoupledKellyHead:
    """Add a separate three-logit action-conditional wager head."""

    def setup(self) -> None:
        super().setup()
        self.kelly_head = nn.Linear(
            self._embedding_dim,
            int(self.action_space.n),
        )

    def _forward_train(self, batch, **kwargs):
        outputs = super()._forward_train(batch, **kwargs)
        outputs[KELLY_LOGITS_KEY] = self.kelly_head(outputs[Columns.EMBEDDINGS])
        return outputs


def realized_log_growth(
    correct: torch.Tensor,
    wager: torch.Tensor,
) -> torch.Tensor:
    """Fair three-way-bet log growth, computed entirely on-device."""

    won = torch.log1p(wager * NET_WIN_ODDS)
    lost = torch.log1p(-wager)
    return torch.where(correct.to(dtype=torch.bool), won, lost)


class PrepareDecoupledKellyBatch(ConnectorV2):
    """Preserve binary correctness before GAE for the direct Kelly loss."""

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
            module_batch[KELLY_CORRECT_KEY] = module_batch[Columns.REWARDS]
        return batch


class DirectKellyLossMixin:
    """Add wager utility without changing PPO's correctness reward."""

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
        correct = batch.get(KELLY_CORRECT_KEY)
        if correct is None:
            return total
        logits = fwd_out[KELLY_LOGITS_KEY]
        actions = batch[Columns.ACTIONS].to(dtype=torch.long)
        wagers = torch.sigmoid(logits).gather(
            -1,
            actions.unsqueeze(-1),
        ).squeeze(-1).clamp(max=MAX_WAGER)
        growth = realized_log_growth(correct > 0.5, wagers)
        mask = batch.get(Columns.LOSS_MASK)
        direct_loss = -_masked_mean(growth, mask)
        coefficient = float(
            config.learner_config_dict.get(
                KELLY_LOSS_COEFFICIENT_KEY,
                0.0,
            )
        )
        if coefficient <= 0.0:
            raise ValueError("Kelly loss coefficient must be positive")
        self.metrics.log_dict(
            {
                f"{KELLY_NAMESPACE}/direct_loss": direct_loss,
                f"{KELLY_NAMESPACE}/log_growth_mean": -direct_loss,
                f"{KELLY_NAMESPACE}/wager_mean": _masked_mean(wagers, mask),
                f"{KELLY_NAMESPACE}/correct_fraction": _masked_mean(
                    correct.to(dtype=wagers.dtype),
                    mask,
                ),
            },
            key=module_id,
            window=1,
        )
        return total + coefficient * direct_loss


class KellyConnectorMixin:
    """Insert correctness capture immediately before PPO's GAE connector."""

    def build(self) -> None:
        super().build()
        self._learner_connector.insert_before(
            GeneralAdvantageEstimation,
            PrepareDecoupledKellyBatch(),
        )


class KellyPPOTorchLearner(
    KellyConnectorMixin,
    DirectKellyLossMixin,
    PPOTorchLearner,
):
    """Correctness PPO plus a direct loss on a separate Kelly head."""
