"""Gamma-zero PPO/IQN composition for coupled and decoupled Kelly credit."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from torch import nn

from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from experiments.mess_3_kelly_cycle_1.kelly import (
    COLLAPSE_THRESHOLD,
    MAX_WAGER,
    realized_log_growth,
)
from experiments.mess_3_kelly_cycle_1.learning import (
    TokenCategoricalWithWager,
    WagerTransformerModel,
)
from learners.models.transformer import TransformerModel


NAMESPACE = "kelly_cycle_2"
ACTOR_MODE_KEY = f"{NAMESPACE}/actor_mode"
WAGER_LAYOUT_KEY = f"{NAMESPACE}/wager_layout"
DIRECT_LOSS_WEIGHT_KEY = f"{NAMESPACE}/direct_loss_weight"
CORRECTNESS_KEY = f"{NAMESPACE}/correct"
BEHAVIOR_WAGER_KEY = f"{NAMESPACE}/behavior_wager"

CORRECTNESS_MODE = "correctness"
COUPLED_MODE = "coupled_kelly"
DECOUPLED_MODE = "decoupled_kelly"
CONDITIONAL_MODE = "conditional_decoupled_kelly"
SCALAR_LAYOUT = "scalar"
CONDITIONAL_LAYOUT = "conditional"
NO_WAGER_LAYOUT = "none"


class TokenCategoricalWithConditionalWager(TorchCategorical):
    """Categorical token distribution carrying one wager logit per token."""

    def __init__(self, logits: torch.Tensor) -> None:
        if logits.shape[-1] % 2:
            raise ValueError("conditional token-plus-wager logits must be even")
        split = logits.shape[-1] // 2
        self.all_logits = logits
        super().__init__(logits=logits[..., :split])

    @staticmethod
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        del kwargs
        if not isinstance(space, gym.spaces.Discrete):
            raise TypeError("conditional wagers require Discrete token actions")
        return int(space.n) * 2


class ConditionalWagerTransformerModel(TransformerModel):
    """Mean-value transformer with one sigmoid wager output per token."""

    def setup(self) -> None:
        super().setup()
        self.kelly_wager_head = nn.Linear(
            self._embedding_dim,
            int(self.action_space.n),
        )
        self.action_dist_cls = TokenCategoricalWithConditionalWager

    def _outputs(
        self,
        embeddings: torch.Tensor,
        state_out: Any | None,
        *,
        training: bool,
    ) -> dict:
        outputs = super()._outputs(
            embeddings,
            state_out,
            training=training,
        )
        outputs[Columns.ACTION_DIST_INPUTS] = torch.cat(
            [
                outputs[Columns.ACTION_DIST_INPUTS],
                self.kelly_wager_head(embeddings),
            ],
            dim=-1,
        )
        return outputs

    def wager_fraction(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.kelly_wager_head(embeddings))


class IQNScalarWagerTransformerModel(IQNTransformerModel):
    """IQN transformer with one state-level sigmoid wager output."""

    def setup(self) -> None:
        super().setup()
        self.kelly_wager_head = nn.Linear(self._embedding_dim, 1)
        self.action_dist_cls = TokenCategoricalWithWager

    def _outputs(
        self,
        embeddings: torch.Tensor,
        state_out: Any | None,
        *,
        training: bool,
    ) -> dict:
        outputs = super()._outputs(
            embeddings,
            state_out,
            training=training,
        )
        outputs[Columns.ACTION_DIST_INPUTS] = torch.cat(
            [
                outputs[Columns.ACTION_DIST_INPUTS],
                self.kelly_wager_head(embeddings),
            ],
            dim=-1,
        )
        return outputs

    def wager_fraction(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.kelly_wager_head(embeddings)).squeeze(-1)


class IQNConditionalWagerTransformerModel(IQNTransformerModel):
    """IQN transformer with one sigmoid wager output per token."""

    def setup(self) -> None:
        super().setup()
        self.kelly_wager_head = nn.Linear(
            self._embedding_dim,
            int(self.action_space.n),
        )
        self.action_dist_cls = TokenCategoricalWithConditionalWager

    def _outputs(
        self,
        embeddings: torch.Tensor,
        state_out: Any | None,
        *,
        training: bool,
    ) -> dict:
        outputs = super()._outputs(
            embeddings,
            state_out,
            training=training,
        )
        outputs[Columns.ACTION_DIST_INPUTS] = torch.cat(
            [
                outputs[Columns.ACTION_DIST_INPUTS],
                self.kelly_wager_head(embeddings),
            ],
            dim=-1,
        )
        return outputs

    def wager_fraction(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.kelly_wager_head(embeddings))


def _valid_mask(batch: dict[str, Any], reference: torch.Tensor) -> torch.Tensor:
    valid = batch.get(Columns.LOSS_MASK)
    if valid is None:
        return torch.ones_like(reference, dtype=torch.bool)
    return valid.to(device=reference.device, dtype=torch.bool)


def selected_wager(
    *,
    action_dist_inputs: torch.Tensor,
    actions: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    """Select behavior wagers from scalar or action-conditional head outputs."""

    action_indices = actions.to(dtype=torch.long)
    if layout == SCALAR_LAYOUT:
        if action_dist_inputs.shape[-1] != 4:
            raise ValueError("scalar wager layout expects four distribution inputs")
        return torch.sigmoid(action_dist_inputs[..., -1]).clamp(max=MAX_WAGER)
    if layout == CONDITIONAL_LAYOUT:
        if action_dist_inputs.shape[-1] != 6:
            raise ValueError(
                "conditional wager layout expects six distribution inputs"
            )
        fractions = torch.sigmoid(action_dist_inputs[..., 3:]).clamp(
            max=MAX_WAGER
        )
        return fractions.gather(
            -1,
            action_indices.unsqueeze(-1),
        ).squeeze(-1)
    raise ValueError(f"unsupported wager layout {layout!r}")


class PrepareKellyBatch(ConnectorV2):
    """Attach wager targets and optionally replace actor rewards with log growth."""

    def __init__(self, *, actor_mode: str, wager_layout: str) -> None:
        super().__init__()
        if actor_mode not in {COUPLED_MODE, DECOUPLED_MODE, CONDITIONAL_MODE}:
            raise ValueError(f"unsupported Kelly actor mode {actor_mode!r}")
        self.actor_mode = actor_mode
        self.wager_layout = wager_layout

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
            correct = module_batch[Columns.REWARDS]
            wager = selected_wager(
                action_dist_inputs=module_batch[Columns.ACTION_DIST_INPUTS],
                actions=module_batch[Columns.ACTIONS],
                layout=self.wager_layout,
            )
            valid = _valid_mask(module_batch, correct)
            growth = realized_log_growth(correct > 0.5, wager)
            module_batch[CORRECTNESS_KEY] = correct
            module_batch[BEHAVIOR_WAGER_KEY] = torch.where(
                valid,
                wager,
                torch.zeros_like(wager),
            )
            if self.actor_mode == COUPLED_MODE:
                module_batch[Columns.REWARDS] = torch.where(
                    valid,
                    growth,
                    torch.zeros_like(growth),
                )
        return batch


def current_selected_wager(
    *,
    action_dist_inputs: torch.Tensor,
    actions: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    """Differentiably select the current policy's wager for each sampled token."""

    return selected_wager(
        action_dist_inputs=action_dist_inputs,
        actions=actions,
        layout=layout,
    )


class DirectKellyLossMixin:
    """Add selected-action Kelly utility while preserving the base actor loss."""

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
        behavior = batch.get(BEHAVIOR_WAGER_KEY)
        correct = batch.get(CORRECTNESS_KEY)
        if behavior is None or correct is None:
            return total

        valid = _valid_mask(batch, behavior)
        weights = valid.to(dtype=behavior.dtype)
        count = weights.sum().clamp_min(1.0)
        current = current_selected_wager(
            action_dist_inputs=fwd_out[Columns.ACTION_DIST_INPUTS],
            actions=batch[Columns.ACTIONS],
            layout=config.learner_config_dict[WAGER_LAYOUT_KEY],
        )
        current_growth = realized_log_growth(correct > 0.5, current)
        direct_loss = -(current_growth * weights).sum() / count
        weight = float(
            config.learner_config_dict.get(DIRECT_LOSS_WEIGHT_KEY, 0.0)
        )
        if weight <= 0.0:
            raise ValueError("Kelly wager arms require a positive direct loss")

        self.metrics.log_dict(
            {
                f"{NAMESPACE}/behavior_wager_mean": (
                    behavior * weights
                ).sum()
                / count,
                f"{NAMESPACE}/behavior_wager_collapse_fraction": (
                    (behavior < COLLAPSE_THRESHOLD).to(behavior.dtype) * weights
                ).sum()
                / count,
                f"{NAMESPACE}/correct_fraction": (
                    correct.to(behavior.dtype) * weights
                ).sum()
                / count,
                f"{NAMESPACE}/current_log_growth_mean": (
                    current_growth * weights
                ).sum()
                / count,
                f"{NAMESPACE}/direct_loss": direct_loss,
            },
            key=module_id,
            window=1,
        )
        return total + weight * direct_loss


class KellyConnectorMixin:
    """Insert the configured Kelly annotation/reward connector before GAE."""

    def build(self) -> None:
        super().build()
        learner_config = self.config.learner_config_dict
        self._learner_connector.insert_before(
            GeneralAdvantageEstimation,
            PrepareKellyBatch(
                actor_mode=learner_config[ACTOR_MODE_KEY],
                wager_layout=learner_config[WAGER_LAYOUT_KEY],
            ),
        )


class KellyMeanPPOTorchLearner(
    KellyConnectorMixin,
    DirectKellyLossMixin,
    PPOTorchLearner,
):
    """Mean-value PPO with direct Kelly wager learning."""


class KellyIQNPPOTorchLearner(
    KellyConnectorMixin,
    DirectKellyLossMixin,
    IQNPPOTorchLearner,
):
    """IQN PPO with direct Kelly wager learning."""


ScalarWagerTransformerModel = WagerTransformerModel
