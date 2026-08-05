"""Small pre-LN causal transformer matching the paper's architecture."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class PaperModelConfig:
    d_vocab: int = 3
    context_length: int = 10
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 1
    d_head: int = 8
    d_mlp: int = 256
    activation: str = "relu"
    normalization: str = "layer_norm"
    positional_embedding: str = "learned_absolute"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SingleHeadCausalAttention(nn.Module):
    def __init__(self, config: PaperModelConfig):
        super().__init__()
        if config.n_heads != 1:
            raise ValueError("the paper replication expects exactly one head")
        self.d_head = config.d_head
        self.query = nn.Linear(config.d_model, config.d_head, bias=True)
        self.key = nn.Linear(config.d_model, config.d_head, bias=True)
        self.value = nn.Linear(config.d_model, config.d_head, bias=True)
        self.output = nn.Linear(config.d_head, config.d_model, bias=True)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.shape[1]
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.d_head)
        scores = scores.masked_fill(
            ~self.causal_mask[:sequence_length, :sequence_length],
            -torch.inf,
        )
        attention = torch.softmax(scores, dim=-1)
        return self.output(torch.matmul(attention, value))


class TransformerBlock(nn.Module):
    def __init__(self, config: PaperModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = SingleHeadCausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.ReLU(),
            nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.mlp(self.mlp_norm(hidden))


class PaperTransformer(nn.Module):
    """TransformerLens-equivalent model with explicit residual access."""

    def __init__(self, config: PaperModelConfig | None = None):
        super().__init__()
        self.config = config or PaperModelConfig()
        self.token_embedding = nn.Embedding(
            self.config.d_vocab,
            self.config.d_model,
        )
        self.position_embedding = nn.Embedding(
            self.config.context_length,
            self.config.d_model,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config)
            for _ in range(self.config.n_layers)
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.unembedding = nn.Linear(
            self.config.d_model,
            self.config.d_vocab,
            bias=True,
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def activation_names(self) -> tuple[str, ...]:
        return (
            *(f"block_{index}" for index in range(self.config.n_layers)),
            "final_ln",
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if tokens.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds the configured context length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        activations: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if return_activations:
                activations[f"block_{index}"] = hidden
        normalized = self.final_norm(hidden)
        if return_activations:
            activations["pre_ln_final"] = hidden
            activations["final_ln"] = normalized
        logits = self.unembedding(normalized)
        if return_activations:
            return logits, activations
        return logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
