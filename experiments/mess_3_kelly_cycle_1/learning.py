"""PPO integrations for policy-dependent Kelly rewards and a wager head."""

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

from experiments.mess_3_kelly_cycle_1.kelly import (
    COLLAPSE_THRESHOLD,
    MAX_WAGER,
    kelly_fraction,
    realized_log_growth,
)
from learners.models.transformer import TransformerModel


NAMESPACE = "kelly"
MODE_KEY = f"{NAMESPACE}/mode"
DIRECT_LOSS_WEIGHT_KEY = f"{NAMESPACE}/direct_loss_weight"
CORRECTNESS_KEY = f"{NAMESPACE}/correct"
BEHAVIOR_WAGER_KEY = f"{NAMESPACE}/behavior_wager"
LEARNED_MODE = "learned_kelly"
FIXED_MODE = "fixed_full"
POLICY_MODE = "policy_implied_kelly"


class TokenCategoricalWithWager(TorchCategorical):
    """Categorical token distribution carrying one non-action wager logit."""

    def __init__(self, logits: torch.Tensor) -> None:
        if logits.shape[-1] < 2:
            raise ValueError("token-plus-wager logits require at least two columns")
        self.all_logits = logits
        super().__init__(logits=logits[..., :-1])

    @staticmethod
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        del kwargs
        if not isinstance(space, gym.spaces.Discrete):
            raise TypeError("token-plus-wager distribution requires Discrete actions")
        return int(space.n) + 1


class WagerTransformerModel(TransformerModel):
    """Transformer whose categorical output also carries a sigmoid wager."""

    def setup(self):
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


def _valid_mask(batch: dict[str, Any], reference: torch.Tensor) -> torch.Tensor:
    valid = batch.get(Columns.LOSS_MASK)
    if valid is None:
        return torch.ones_like(reference, dtype=torch.bool)
    return valid.to(device=reference.device, dtype=torch.bool)


def behavior_wager(
    *,
    mode: str,
    action_dist_inputs: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Compute detached rollout wagers without leaving the training device."""

    token_logits = action_dist_inputs[..., :3]
    if mode == FIXED_MODE:
        return torch.full_like(actions, MAX_WAGER, dtype=token_logits.dtype)
    if mode == POLICY_MODE:
        probabilities = torch.softmax(token_logits, dim=-1)
        selected = probabilities.gather(
            -1,
            actions.to(dtype=torch.long).unsqueeze(-1),
        ).squeeze(-1)
        return kelly_fraction(selected)
    if mode == LEARNED_MODE:
        if action_dist_inputs.shape[-1] != 4:
            raise ValueError("learned Kelly mode requires one wager logit")
        return torch.sigmoid(action_dist_inputs[..., -1]).clamp(max=MAX_WAGER)
    raise ValueError(f"unsupported learner-side Kelly mode {mode!r}")


class FormKellyRewards(ConnectorV2):
    """Replace correctness rewards with detached behavior-policy log growth."""

    def __init__(self, *, mode: str) -> None:
        super().__init__()
        if mode not in {FIXED_MODE, POLICY_MODE, LEARNED_MODE}:
            raise ValueError(f"unsupported Kelly reward mode {mode!r}")
        self.mode = mode

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
            wager = behavior_wager(
                mode=self.mode,
                action_dist_inputs=module_batch[Columns.ACTION_DIST_INPUTS],
                actions=module_batch[Columns.ACTIONS],
            )
            valid = _valid_mask(module_batch, correct)
            growth = realized_log_growth(correct > 0.5, wager)
            module_batch[CORRECTNESS_KEY] = correct
            module_batch[BEHAVIOR_WAGER_KEY] = torch.where(
                valid,
                wager,
                torch.zeros_like(wager),
            )
            module_batch[Columns.REWARDS] = torch.where(
                valid,
                growth,
                torch.zeros_like(growth),
            )
        return batch


class KellyObjectiveMixin:
    """Log wager diagnostics and optionally train the deterministic wager head."""

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
        behavior_mean = (behavior * weights).sum() / count
        collapse = (
            (behavior < COLLAPSE_THRESHOLD).to(behavior.dtype) * weights
        ).sum() / count
        metrics = {
            f"{NAMESPACE}/behavior_wager_mean": behavior_mean,
            f"{NAMESPACE}/behavior_wager_collapse_fraction": collapse,
            f"{NAMESPACE}/correct_fraction": (
                correct.to(behavior.dtype) * weights
            ).sum()
            / count,
            f"{NAMESPACE}/log_growth_mean": (
                batch[Columns.REWARDS] * weights
            ).sum()
            / count,
        }

        weight = float(
            config.learner_config_dict.get(DIRECT_LOSS_WEIGHT_KEY, 0.0)
        )
        inputs = fwd_out[Columns.ACTION_DIST_INPUTS]
        if weight > 0.0:
            if inputs.shape[-1] != 4:
                raise ValueError("direct Kelly loss requires one wager logit")
            current_wager = torch.sigmoid(inputs[..., -1]).clamp(max=MAX_WAGER)
            per_item_growth = realized_log_growth(
                correct > 0.5,
                current_wager,
            )
            direct_loss = -(per_item_growth * weights).sum() / count
            metrics.update(
                {
                    f"{NAMESPACE}/direct_loss": direct_loss,
                    f"{NAMESPACE}/current_wager_mean": (
                        current_wager * weights
                    ).sum()
                    / count,
                }
            )
            total = total + weight * direct_loss

        self.metrics.log_dict(metrics, key=module_id, window=1)
        return total


class KellyRewardPPOTorchLearner(KellyObjectiveMixin, PPOTorchLearner):
    """PPO using Kelly log growth formed from behavior-policy outputs."""

    def build(self) -> None:
        super().build()
        mode = self.config.learner_config_dict.get(MODE_KEY)
        self._learner_connector.insert_before(
            GeneralAdvantageEstimation,
            FormKellyRewards(mode=mode),
        )
