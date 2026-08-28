"""A 64-dimensional paper-style transformer adapted to stateful RLlib PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.core.columns import Columns
from torch import nn

from learners.models.base import BaseActorCriticModel


@dataclass(frozen=True, slots=True)
class FactoredReproductionModelConfig:
    """The paper's canonical architecture, reduced only to ``d_model=64``.

    Four heads are used because the paper's three 40-dimensional heads cannot
    divide a 64-dimensional residual stream. The resulting head width is 16.
    """

    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 256
    context_length: int = 9
    max_seq_len: int = 32
    activation: str = "relu"
    normalization: str = "layer_norm"
    positional_embedding: str = "learned_absolute"

    def __post_init__(self) -> None:
        if min(
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_mlp,
            self.context_length,
            self.max_seq_len,
        ) <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.activation != "relu":
            raise ValueError("the paper architecture uses ReLU")
        if self.normalization != "layer_norm":
            raise ValueError("the paper architecture uses LayerNorm")
        if self.positional_embedding != "learned_absolute":
            raise ValueError("the paper architecture uses learned positions")

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> FactoredReproductionModelConfig:
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiHeadCausalAttention(nn.Module):
    """Standard dense multi-head causal attention with a supplied validity mask."""

    def __init__(self, config: FactoredReproductionModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch,
            length,
            3,
            self.n_heads,
            self.d_head,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed[:, None, :, :],
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class ReproductionTransformerBlock(nn.Module):
    """Pre-LayerNorm attention and ReLU MLP residual block."""

    def __init__(self, config: FactoredReproductionModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadCausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.ReLU(),
            nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs), allowed)
        return hidden + self.mlp(self.mlp_norm(hidden))


class ReproductionResidualEncoder(nn.Module):
    """Strict-window encoder exposing the paper's pre-final-LN residual stream."""

    def __init__(
        self,
        obs_dim: int,
        config: FactoredReproductionModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.input_embedding = nn.Linear(obs_dim, config.d_model, bias=False)
        self.bos_embedding = nn.Parameter(torch.empty(config.d_model))
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            ReproductionTransformerBlock(config)
            for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.apply(self._initialize)
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def token_embedding_matrix(self) -> torch.Tensor:
        """Return one row per visible joint-token embedding, excluding BOS."""

        return self.input_embedding.weight.transpose(0, 1)

    def forward(
        self,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        observations: torch.Tensor,
        *,
        apply_final_norm: bool = True,
    ) -> torch.Tensor:
        if observations.ndim != 3:
            raise ValueError("observations must have shape (B, T, D)")
        batch_size, steps, width = observations.shape
        if width != self.obs_dim:
            raise ValueError("observation width does not match the encoder")
        window = self.config.context_length
        combined = torch.cat([context, observations], dim=1)
        windows = combined.unfold(1, window, 1).permute(0, 1, 3, 2)
        if windows.shape[1] != steps:
            raise RuntimeError("context state does not match configured window")
        flat_windows = windows.reshape(batch_size * steps, window, width)

        offsets = torch.arange(steps, device=observations.device)
        valid_lengths = (
            context_lengths.reshape(-1, 1).to(dtype=torch.long)
            + offsets.reshape(1, -1)
            + 1
        ).clamp(max=window)
        valid_lengths = valid_lengths.reshape(-1)
        slots = torch.arange(window, device=observations.device)
        first_valid = window - valid_lengths
        valid = slots.reshape(1, -1) >= first_valid.reshape(-1, 1)
        positions = (slots.reshape(1, -1) - first_valid.reshape(-1, 1)).clamp_min(
            0
        )

        hidden = self.input_embedding(flat_windows)
        is_bos = valid & (flat_windows.abs().sum(dim=-1) < 0.5)
        hidden = hidden + is_bos.unsqueeze(-1).to(hidden.dtype) * self.bos_embedding
        hidden = hidden + self.position_embedding(positions)

        causal = slots.reshape(-1, 1) >= slots.reshape(1, -1)
        allowed = causal.reshape(1, window, window) & valid.reshape(
            -1,
            1,
            window,
        )
        diagonal = torch.eye(
            window,
            dtype=torch.bool,
            device=observations.device,
        ).reshape(1, window, window)
        allowed = allowed | diagonal
        for block in self.blocks:
            hidden = block(hidden, allowed)
        if apply_final_norm:
            hidden = self.final_norm(hidden)
        encoded = hidden[:, -1, :]
        return encoded.reshape(batch_size, steps, self.config.d_model)


class FactoredReproductionActorCritic(BaseActorCriticModel):
    """Stateful actor-critic whose probe representation matches Appendix F."""

    def _build_encoder(self) -> int:
        self.reproduction_config = FactoredReproductionModelConfig.from_dict(
            dict(self.model_config)
        )
        self._obs_dim = int(np.prod(self.observation_space.shape))
        self.encoder = ReproductionResidualEncoder(
            self._obs_dim,
            self.reproduction_config,
        )
        return self.reproduction_config.d_model

    @property
    def sequence_lookback(self) -> int:
        return self.reproduction_config.context_length - 1

    def get_initial_state(self) -> dict[str, np.ndarray]:
        return {
            "ctx": np.zeros(
                (self.sequence_lookback, self._obs_dim),
                dtype=np.float32,
            ),
            "len": np.zeros((1,), dtype=np.float32),
        }

    def _advance_context(
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

    def _encode_with_norm(
        self,
        batch: dict[str, Any],
        *,
        apply_final_norm: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = batch[Columns.OBS]
        state = batch[Columns.STATE_IN]
        embeddings = self.encoder(
            state["ctx"],
            state["len"].reshape(-1),
            observations,
            apply_final_norm=apply_final_norm,
        )
        return embeddings, self._advance_context(observations, state)

    def _encode_train(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._encode_with_norm(batch, apply_final_norm=True)

    def _encode_rollout(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._encode_with_norm(batch, apply_final_norm=True)

    @torch.no_grad()
    def encode_step(
        self,
        observation: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings, state_out = self._encode_with_norm(
            {
                Columns.OBS: observation.unsqueeze(1),
                Columns.STATE_IN: state,
            },
            apply_final_norm=True,
        )
        return embeddings[:, 0, :], state_out

    @torch.no_grad()
    def encode_step_pre_final_norm(
        self,
        observation: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings, state_out = self._encode_with_norm(
            {
                Columns.OBS: observation.unsqueeze(1),
                Columns.STATE_IN: state,
            },
            apply_final_norm=False,
        )
        return embeddings[:, 0, :], state_out

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
