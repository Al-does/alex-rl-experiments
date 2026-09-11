"""A configurable paper-style transformer adapted to stateful RLlib PPO."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from torch import nn

from learners.components.transformer import _apply_rope, _rope_angles
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
        if self.activation not in {"relu", "gated_gelu"}:
            raise ValueError("activation must be relu or gated_gelu")
        if self.normalization not in {"layer_norm", "rms_norm"}:
            raise ValueError("normalization must be layer_norm or rms_norm")
        if self.positional_embedding not in {"learned_absolute", "rope"}:
            raise ValueError("positional_embedding must be learned_absolute or rope")
        if self.positional_embedding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")

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
        self.use_rope = config.positional_embedding == "rope"
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
        if self.use_rope:
            cos, sin = _rope_angles(length, self.d_head, query.device, query.dtype)
            query, key = _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.d_head)
        scores = scores.masked_fill(~allowed[:, None, :, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class GatedGELUMLP(nn.Module):
    def __init__(self, config: FactoredReproductionModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_mlp)
        self.value = nn.Linear(config.d_model, config.d_mlp)
        self.output = nn.Linear(config.d_mlp, config.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.gelu(self.gate(inputs))
        return self.output(gated * self.value(inputs))


def _normalization(
    config: FactoredReproductionModelConfig,
) -> nn.LayerNorm | nn.RMSNorm:
    if config.normalization == "rms_norm":
        return nn.RMSNorm(config.d_model, eps=1e-5)
    return nn.LayerNorm(config.d_model)


def _mlp(config: FactoredReproductionModelConfig) -> nn.Module:
    if config.activation == "gated_gelu":
        return GatedGELUMLP(config)
    return nn.Sequential(
        nn.Linear(config.d_model, config.d_mlp),
        nn.ReLU(),
        nn.Linear(config.d_mlp, config.d_model),
    )


class ReproductionTransformerBlock(nn.Module):
    """Pre-normalization attention and MLP residual block."""

    def __init__(self, config: FactoredReproductionModelConfig) -> None:
        super().__init__()
        self.attention_norm = _normalization(config)
        self.attention = MultiHeadCausalAttention(config)
        self.mlp_norm = _normalization(config)
        self.mlp = _mlp(config)

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
        self.position_embedding = (
            nn.Embedding(config.context_length, config.d_model)
            if config.positional_embedding == "learned_absolute"
            else None
        )
        self.blocks = nn.ModuleList(
            ReproductionTransformerBlock(config)
            for _ in range(config.n_layers)
        )
        self.final_norm = _normalization(config)
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
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

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
        if self.position_embedding is not None:
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

    _VALUE_CHUNK_STEPS = 4096

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

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: torch.Tensor | None = None,
    ):
        if embeddings is not None:
            return self.heads.values(embeddings)
        with torch.no_grad():
            observations = batch[Columns.OBS]
            state_in = batch[Columns.STATE_IN]
            steps = int(observations.shape[1])
            rows_per_chunk = max(1, self._VALUE_CHUNK_STEPS // steps)
            if int(observations.shape[0]) <= rows_per_chunk:
                embeddings, _ = self._encode_train(batch)
                return self.heads.values(embeddings)
            values = []
            for start in range(0, int(observations.shape[0]), rows_per_chunk):
                stop = start + rows_per_chunk
                chunk = {
                    Columns.OBS: observations[start:stop],
                    Columns.STATE_IN: {
                        key: value[start:stop] for key, value in state_in.items()
                    },
                }
                chunk_embeddings, _ = self._encode_train(chunk)
                values.append(self.heads.values(chunk_embeddings))
            return torch.cat(values, dim=0)

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
