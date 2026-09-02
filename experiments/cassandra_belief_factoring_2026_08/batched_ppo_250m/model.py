"""Batched actor-critic model with device-resident rollout state."""

from __future__ import annotations

from dataclasses import dataclass

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
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_count = action_count
        self.encoder = CausalTransformerEncoder(
            obs_dim=observation_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            context_len=context_len,
        )
        self.policy = nn.Linear(d_model, action_count)
        self.value = nn.Linear(d_model, 1)

    def initial_state(
        self, batch_size: int, device: torch.device
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
            kv_k=torch.zeros(cache_shape, device=device),
            kv_v=torch.zeros(cache_shape, device=device),
            kv_len=torch.zeros(batch_size, device=device),
            raw_context=torch.zeros(
                (
                    batch_size,
                    encoder.lookback,
                    self.observation_dim,
                ),
                device=device,
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
        embeddings, kv_k, kv_v, kv_len = self.encoder.forward_cached(
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
        embeddings = self.encoder(
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
    "InferenceState",
]
