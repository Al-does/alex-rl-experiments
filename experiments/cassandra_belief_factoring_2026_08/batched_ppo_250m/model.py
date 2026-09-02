"""Batched actor-critic model with device-resident rollout state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from learners.components.transformer import CausalTransformerEncoder


@dataclass
class InferenceState:
    kv_k: torch.Tensor
    kv_v: torch.Tensor
    kv_len: torch.Tensor
    raw_context: torch.Tensor
    context_position: int
    context_length: int


def _apply_rope(
    values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    even, odd = values[..., 0::2], values[..., 1::2]
    output = torch.empty_like(values)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output


class CachedRolloutTransformerEncoder(CausalTransformerEncoder):
    """Cache device-static RoPE and masking tensors used at every rollout step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        half = self.head_dim // 2
        inverse_frequency = 1.0 / (
            10000.0
            ** (torch.arange(half, dtype=torch.float32) / half)
        )
        query_position = self.lookback
        key_positions = torch.arange(
            query_position - self.context_len,
            query_position + 1,
            dtype=torch.float32,
        )
        key_angles = key_positions[:, None] * inverse_frequency[None, :]
        query_angles = (
            torch.tensor([query_position], dtype=torch.float32)[:, None]
            * inverse_frequency[None, :]
        )
        self.register_buffer(
            "_rollout_cos_k", torch.cos(key_angles), persistent=False
        )
        self.register_buffer(
            "_rollout_sin_k", torch.sin(key_angles), persistent=False
        )
        self.register_buffer(
            "_rollout_cos_q", torch.cos(query_angles), persistent=False
        )
        self.register_buffer(
            "_rollout_sin_q", torch.sin(query_angles), persistent=False
        )
        self.register_buffer(
            "_rollout_slots",
            torch.arange(self.cache_len),
            persistent=False,
        )

    def forward_cached(
        self,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        kv_len: torch.Tensor,
        obs: torch.Tensor,
        *,
        apply_final_norm: bool = True,
    ):
        batch, chunk_len, _ = obs.shape
        embeddings = []
        for timestep in range(chunk_len):
            x_t = self.input_projection(obs[:, timestep : timestep + 1])
            kv_len = torch.clamp(
                kv_len + 1.0, max=float(self.cache_len)
            )
            valid = (
                self._rollout_slots[None, :]
                >= (self.cache_len - kv_len[:, None])
            ).view(batch, 1, 1, self.cache_len)
            next_k, next_v = [], []
            for layer, block in enumerate(self.blocks):
                _, _, width = x_t.shape
                q, k, v = block.qkv(block.ln1(x_t)).chunk(3, dim=-1)

                def split(tensor):
                    return tensor.view(
                        batch,
                        1,
                        block.n_heads,
                        block.head_dim,
                    ).transpose(1, 2)

                q, k, v = split(q), split(k), split(v)
                k_cache = torch.cat(
                    [kv_k[:, layer, :, 1:], k], dim=2
                )
                v_cache = torch.cat(
                    [kv_v[:, layer, :, 1:], v], dim=2
                )
                q = _apply_rope(
                    q, self._rollout_cos_q, self._rollout_sin_q
                )
                rotated_k = _apply_rope(
                    k_cache,
                    self._rollout_cos_k,
                    self._rollout_sin_k,
                )
                attention = (
                    q @ rotated_k.transpose(-2, -1)
                ) / math.sqrt(block.head_dim)
                attention = attention.masked_fill(
                    ~valid, float("-inf")
                )
                attention = attention.softmax(dim=-1) @ v_cache
                attention = attention.transpose(1, 2).reshape(
                    batch, 1, width
                )
                x_t = x_t + block.proj(attention)
                x_t = x_t + block.mlp(block.ln2(x_t))
                next_k.append(k_cache)
                next_v.append(v_cache)
            kv_k = torch.stack(next_k, dim=1)
            kv_v = torch.stack(next_v, dim=1)
            embeddings.append(
                self.final_norm(x_t) if apply_final_norm else x_t
            )
        return torch.cat(embeddings, dim=1), kv_k, kv_v, kv_len


class BatchedTransformerActorCritic(nn.Module):
    """The original Cassandra transformer without RLlib state serialization."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_count: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        context_len: int,
        cache_inference_constants: bool = False,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_count = action_count
        encoder_class = (
            CachedRolloutTransformerEncoder
            if cache_inference_constants
            else CausalTransformerEncoder
        )
        self.encoder = encoder_class(
            obs_dim=observation_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            context_len=context_len,
        )
        self.policy = nn.Linear(d_model, action_count)
        self.value = nn.Linear(d_model, 1)
        self._rollout_encoder = self.encoder.forward_cached
        self._training_encoder = self.encoder.forward

    def enable_compilation(self) -> None:
        self._rollout_encoder = torch.compile(
            self.encoder.forward_cached,
            mode="reduce-overhead",
        )
        self._training_encoder = torch.compile(
            self.encoder.forward,
            mode="reduce-overhead",
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> InferenceState:
        encoder = self.encoder
        cache_shape = (
            batch_size,
            encoder.n_layers,
            encoder.n_heads,
            encoder.cache_len,
            encoder.head_dim,
        )
        return InferenceState(
            kv_k=torch.zeros(cache_shape, device=device, dtype=dtype),
            kv_v=torch.zeros(cache_shape, device=device, dtype=dtype),
            kv_len=torch.zeros(batch_size, device=device),
            raw_context=torch.zeros(
                (
                    batch_size,
                    encoder.lookback,
                    self.observation_dim,
                ),
                device=device,
                dtype=dtype,
            ),
            context_position=0,
            context_length=0,
        )

    def inference(
        self,
        observations: torch.Tensor,
        state: InferenceState,
        *,
        record_context: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, InferenceState]:
        embeddings, kv_k, kv_v, kv_len = self._rollout_encoder(
            state.kv_k,
            state.kv_v,
            state.kv_len,
            observations[:, None, :],
        )
        position = state.context_position
        if record_context:
            state.raw_context[:, position].copy_(observations)
        next_state = InferenceState(
            kv_k=kv_k,
            kv_v=kv_v,
            kv_len=kv_len,
            raw_context=state.raw_context,
            context_position=(
                (position + 1) % self.encoder.lookback
                if record_context
                else position
            ),
            context_length=(
                min(state.context_length + 1, self.encoder.lookback)
                if record_context
                else state.context_length
            ),
        )
        embeddings = embeddings[:, 0]
        return (
            self.policy(embeddings),
            self.value(embeddings).squeeze(-1),
            next_state,
        )

    def training_outputs(
        self,
        contexts: torch.Tensor,
        context_lengths: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._training_encoder(
            contexts,
            context_lengths,
            observations,
        )
        return self.policy(embeddings), self.value(embeddings).squeeze(-1)

    @staticmethod
    def ordered_context(state: InferenceState) -> torch.Tensor:
        """Materialize the ring buffer with left padding, once per rollout."""

        context = state.raw_context
        lookback = context.shape[1]
        columns = torch.arange(lookback, device=context.device)
        indices = (state.context_position + columns - lookback) % lookback
        ordered = context[:, indices].clone()
        padding = lookback - state.context_length
        if padding:
            ordered[:, :padding].zero_()
        return ordered


__all__ = [
    "BatchedTransformerActorCritic",
    "CachedRolloutTransformerEncoder",
    "InferenceState",
]
