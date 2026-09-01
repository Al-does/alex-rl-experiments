"""Cycle-5-style RoPE transformers for discrete SAC actor and critics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.torch.base import TorchModel

from experiments.factored_representations_reproduction_SAC_2026_08.model import (
    FactoredReproductionSAC,
    ReproductionSACCatalog,
    ReproductionSACEncoder,
    ReproductionSACEncoderConfig,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    JOINT_TOKEN_COUNT,
    LOCAL_CONTEXT_LENGTH,
    SAC_HISTORY_LENGTH,
    TRANSFORMER_LOOKBACK,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import N_ACTIONS
from learners.components.transformer import CausalTransformerEncoder
from learners.models.transformer import TransformerModelConfig


FRAME_WIDTH = JOINT_TOKEN_COUNT + N_ACTIONS
FLAT_OBSERVATION_WIDTH = SAC_HISTORY_LENGTH * FRAME_WIDTH


@dataclass
class TwoFactorRoPESACEncoderConfig(ReproductionSACEncoderConfig):
    @property
    def output_dims(self) -> tuple[int, ...]:
        return (TransformerModelConfig.from_dict(self.transformer).d_model,)

    def build(self, framework: str) -> TwoFactorRoPESACEncoder:
        if framework != "torch":
            raise ValueError("the two-factor SAC study supports only PyTorch")
        return TwoFactorRoPESACEncoder(self)


class TwoFactorRoPESACEncoder(ReproductionSACEncoder):
    """Encode a fixed history with cycle 5's banded RoPE transformer."""

    def __init__(self, config: TwoFactorRoPESACEncoderConfig) -> None:
        # Skip ReproductionSACEncoder's learned-absolute encoder construction.
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.transformer_config = TransformerModelConfig.from_dict(
            config.transformer
        )
        if self.transformer_config.context_len != LOCAL_CONTEXT_LENGTH:
            raise ValueError("local attention context must match cycle 5")
        self.token_count = FRAME_WIDTH
        self.encoder = CausalTransformerEncoder(
            obs_dim=FRAME_WIDTH,
            d_model=self.transformer_config.d_model,
            n_layers=self.transformer_config.n_layers,
            n_heads=self.transformer_config.n_heads,
            context_len=self.transformer_config.context_len,
        )
        if self.encoder.lookback != TRANSFORMER_LOOKBACK:
            raise ValueError("SAC history must cover the transformer lookback")

    @staticmethod
    def _frames(observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != FLAT_OBSERVATION_WIDTH:
            raise ValueError(
                f"expected observation width {FLAT_OBSERVATION_WIDTH}, "
                f"received {observations.shape[-1]}"
            )
        flat = observations.reshape(-1, FLAT_OBSERVATION_WIDTH)
        token_end = SAC_HISTORY_LENGTH * JOINT_TOKEN_COUNT
        tokens = flat[:, :token_end].reshape(
            -1, SAC_HISTORY_LENGTH, JOINT_TOKEN_COUNT
        )
        actions = flat[:, token_end:].reshape(
            -1, SAC_HISTORY_LENGTH, N_ACTIONS
        )
        return torch.cat((tokens, actions), dim=-1).flip(dims=(-2,))

    def encode_pre_final_norm(self, observations: torch.Tensor) -> torch.Tensor:
        prefix = observations.shape[:-1]
        frames = self._frames(observations)
        context = frames[:, :-1, :]
        current = frames[:, -1:, :]
        context_lengths = (
            context[..., :JOINT_TOKEN_COUNT].abs().sum(dim=-1) > 0.5
        ).sum(dim=-1)
        residual = self.encoder(
            context,
            context_lengths,
            current,
            apply_final_norm=False,
        )[:, 0, :]
        return residual.reshape(*prefix, self.transformer_config.d_model)

    def _forward(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        residual = self.encode_pre_final_norm(inputs[Columns.OBS].float())
        return {ENCODER_OUT: self.encoder.final_norm(residual)}


class TwoFactorRoPESACCatalog(ReproductionSACCatalog):
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
        if tuple(observation_space.shape) != (FLAT_OBSERVATION_WIDTH,):
            raise ValueError("observation does not match the cycle-2 history")
        resolved = TransformerModelConfig.from_dict(model_config_dict)
        self.token_count = FRAME_WIDTH
        self.latent_dims = (resolved.d_model,)
        self.qf_head_config.input_dims = self.latent_dims

    def _make_encoder_config(
        self, *, actor: bool
    ) -> TwoFactorRoPESACEncoderConfig:
        return TwoFactorRoPESACEncoderConfig(
            input_dims=tuple(self.observation_space.shape),
            token_count=self.token_count,
            transformer=dict(self._model_config_dict),
            actor=actor,
            auxiliary_classes=None,
        )


class TwoFactorRewardSAC(FactoredReproductionSAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("catalog_class", TwoFactorRoPESACCatalog)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def actor_hidden(self, observations: torch.Tensor) -> torch.Tensor:
        return self.pi_encoder.encode_pre_final_norm(observations)
