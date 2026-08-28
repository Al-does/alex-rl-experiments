"""Fully split transformer actor and critics for RLlib discrete SAC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.sac.sac_learner import (
    ACTION_LOG_PROBS,
    ACTION_LOG_PROBS_NEXT,
    ACTION_PROBS,
    ACTION_PROBS_NEXT,
    QF_PREDS,
    QF_TARGET_NEXT,
    QF_TWIN_PREDS,
)
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
    DefaultSACTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import MLPHeadConfig, ModelConfig
from ray.rllib.core.models.torch.base import TorchModel

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
    ReproductionResidualEncoder,
)
from experiments.factored_representations_reproduction_SAC_2026_08.process import (
    CONTEXT_LENGTH,
)
from losses.next_token import FWD_KEY


@dataclass
class ReproductionSACEncoderConfig(ModelConfig):
    """Configuration for one independent SAC transformer encoder."""

    token_count: int = 0
    transformer: dict[str, Any] = field(default_factory=dict)
    actor: bool = False
    auxiliary_classes: int | None = None

    @property
    def output_dims(self) -> tuple[int, ...]:
        return (FactoredReproductionModelConfig.from_dict(self.transformer).d_model,)

    def build(self, framework: str) -> ReproductionSACEncoder:
        if framework != "torch":
            raise ValueError("the reproduction supports only PyTorch")
        return ReproductionSACEncoder(self)


class ReproductionSACEncoder(TorchModel, Encoder):
    """Encode a fixed token-history observation with one paper transformer."""

    def __init__(self, config: ReproductionSACEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.reproduction_config = FactoredReproductionModelConfig.from_dict(
            config.transformer
        )
        if self.reproduction_config.context_length != CONTEXT_LENGTH:
            raise ValueError("SAC history depth must equal transformer context length")
        if config.token_count <= 0:
            raise ValueError("token_count must be positive")
        self.token_count = config.token_count
        self.encoder = ReproductionResidualEncoder(
            config.token_count,
            self.reproduction_config,
        )
        if config.actor and config.auxiliary_classes is not None:
            if config.auxiliary_classes <= 0:
                raise ValueError("auxiliary_classes must be positive")
            self.next_token_aux_head = torch.nn.Linear(
                self.reproduction_config.d_model,
                config.auxiliary_classes,
            )

    def _pre_final_norm(self, observations: torch.Tensor) -> torch.Tensor:
        expected = CONTEXT_LENGTH * self.token_count
        if observations.shape[-1] != expected:
            raise ValueError(
                f"expected flattened history width {expected}, "
                f"received {observations.shape[-1]}"
            )
        prefix = observations.shape[:-1]
        newest_first = observations.reshape(-1, CONTEXT_LENGTH, self.token_count)
        oldest_first = newest_first.flip(dims=(-2,))
        visible_count = (
            oldest_first.abs().sum(dim=-1) > 0.5
        ).sum(dim=-1)
        residual = self.encoder(
            oldest_first[:, :-1, :],
            visible_count,
            oldest_first[:, -1:, :],
            apply_final_norm=False,
        )[:, 0, :]
        return residual.reshape(*prefix, self.reproduction_config.d_model)

    def encode_pre_final_norm(self, observations: torch.Tensor) -> torch.Tensor:
        """Return this encoder's final-block residual before its final LayerNorm."""

        return self._pre_final_norm(observations)

    def _forward(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        residual = self._pre_final_norm(inputs[Columns.OBS].float())
        return {ENCODER_OUT: self.encoder.final_norm(residual)}


class ReproductionSACCatalog(SACCatalog):
    """Build independent actor, critic, and twin-critic transformers."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict[str, Any],
        view_requirements: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            model_config_dict,
            view_requirements,
        )
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("the SAC transformer expects a flat Box observation")
        width = int(np.prod(observation_space.shape))
        if width % CONTEXT_LENGTH:
            raise ValueError("observation width must divide into nine token slots")
        self.token_count = width // CONTEXT_LENGTH
        self.reproduction_config = FactoredReproductionModelConfig.from_dict(
            model_config_dict
        )
        self.latent_dims = (self.reproduction_config.d_model,)
        self.pi_and_qf_head_hiddens = []
        self.qf_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=[],
            output_layer_dim=action_space.n,
            output_layer_activation="linear",
        )

    def _make_encoder_config(self, *, actor: bool) -> ReproductionSACEncoderConfig:
        auxiliary = self._model_config_dict.get("next_token_aux", {})
        return ReproductionSACEncoderConfig(
            input_dims=tuple(self.observation_space.shape),
            token_count=self.token_count,
            transformer=dict(self._model_config_dict),
            actor=actor,
            auxiliary_classes=(
                int(auxiliary["num_classes"])
                if actor and "num_classes" in auxiliary
                else None
            ),
        )

    def build_encoder(self, framework: str) -> Encoder:
        return self._make_encoder_config(actor=True).build(framework)

    def build_qf_encoder(self, framework: str) -> Encoder:
        return self._make_encoder_config(actor=False).build(framework)


class FactoredReproductionSAC(DefaultSACTorchRLModule):
    """RLlib discrete SAC with no parameters shared across actor and critics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("catalog_class", ReproductionSACCatalog)
        super().__init__(*args, **kwargs)

    def _forward_train_discrete(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

        actor_next = self.pi_encoder(batch_next)[ENCODER_OUT]
        next_logits = self.pi(actor_next)
        next_log_probs = F.log_softmax(next_logits, dim=-1)
        output = {
            ACTION_PROBS_NEXT: next_log_probs.exp(),
            ACTION_LOG_PROBS_NEXT: next_log_probs,
            QF_TARGET_NEXT: self.forward_target(batch_next, squeeze=False),
            QF_PREDS: self._qf_forward_train_helper(
                batch_curr,
                self.qf_encoder,
                self.qf,
                squeeze=False,
            ),
        }
        if self.twin_q:
            output[QF_TWIN_PREDS] = self._qf_forward_train_helper(
                batch_curr,
                self.qf_twin_encoder,
                self.qf_twin,
                squeeze=False,
            )

        actor_current = self.pi_encoder(batch_curr)[ENCODER_OUT]
        current_logits = self.pi(actor_current)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        output[Columns.EMBEDDINGS] = actor_current
        output[ACTION_PROBS] = current_log_probs.exp()
        output[ACTION_LOG_PROBS] = current_log_probs
        auxiliary_head = getattr(self.pi_encoder, "next_token_aux_head", None)
        if auxiliary_head is not None:
            output[FWD_KEY] = auxiliary_head(actor_current)
        return output

    @property
    def encoder(self) -> ReproductionResidualEncoder:
        """Actor encoder alias used by the shared, algorithm-neutral probes."""

        return self.pi_encoder.encoder

    @property
    def reproduction_config(self) -> FactoredReproductionModelConfig:
        return self.pi_encoder.reproduction_config

    @property
    def sequence_lookback(self) -> int:
        return self.reproduction_config.context_length - 1

    def get_actor_probe_initial_state(self) -> dict[str, np.ndarray]:
        """State used only by on-demand probes over one-token observations."""

        return {
            "ctx": np.zeros(
                (self.sequence_lookback, self.pi_encoder.token_count),
                dtype=np.float32,
            ),
            "len": np.zeros((1,), dtype=np.float32),
        }

    def _advance_probe_context(
        self,
        observations: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        sequence = torch.cat([state["ctx"], observations], dim=1)
        lengths = state["len"].reshape(-1) + observations.shape[1]
        return {
            "ctx": sequence[:, -self.sequence_lookback :, :],
            "len": lengths.clamp(max=self.sequence_lookback).reshape(-1, 1),
        }

    @torch.no_grad()
    def encode_step_pre_final_norm(
        self,
        observation: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = observation.unsqueeze(1)
        residual = self.encoder(
            state["ctx"],
            state["len"].reshape(-1),
            observations,
            apply_final_norm=False,
        )[:, 0, :]
        return residual, self._advance_probe_context(observations, state)

    def encode_chunks_pre_final_norm(
        self,
        context: torch.Tensor,
        lengths: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            context,
            lengths,
            observations,
            apply_final_norm=False,
        )

    def action_distribution_inputs(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.pi(embeddings)
