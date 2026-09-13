from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .process import CONTEXT_LENGTH, VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_vocab: int = VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 512
    initialization_std: float = 0.02
    normalization_epsilon: float = 1e-5
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if min(
            self.d_vocab,
            self.context_length,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_mlp,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dimension % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.initialization_std <= 0.0:
            raise ValueError("initialization_std must be positive")

    @property
    def head_dimension(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "head_dimension": self.head_dimension,
            "activation": "gated_gelu",
            "normalization": "rms_norm",
            "positional_embedding": "rope",
            "attention_implementation": "scaled_dot_product_attention",
        }


def _apply_rope(
    inputs: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    even, odd = inputs[..., 0::2], inputs[..., 1::2]
    output = torch.empty_like(inputs)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dimension = config.head_dimension
        self.rope_base = config.rope_base
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, length, width = inputs.shape
        query, key, value = (
            self.qkv(inputs)
            .reshape(
                batch_size,
                length,
                3,
                self.n_heads,
                self.head_dimension,
            )
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        half = self.head_dimension // 2
        inverse_frequencies = 1.0 / (
            self.rope_base
            ** (
                torch.arange(
                    half,
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
                / half
            )
        )
        angles = (
            torch.arange(length, device=inputs.device, dtype=inputs.dtype)[
                :, None
            ]
            * inverse_frequencies[None, :]
        )
        cosine, sine = torch.cos(angles), torch.sin(angles)
        query = _apply_rope(query, cosine, sine)
        key = _apply_rope(key, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        return self.output(
            attended.transpose(1, 2).reshape(batch_size, length, width)
        )


class GatedGELUMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(config.d_model, config.d_mlp)
        self.gate = nn.Linear(config.d_model, config.d_mlp)
        self.output = nn.Linear(config.d_mlp, config.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.gate(inputs)) * self.input(inputs))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(
            config.d_model,
            eps=config.normalization_epsilon,
        )
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.RMSNorm(
            config.d_model,
            eps=config.normalization_epsilon,
        )
        self.mlp = GatedGELUMLP(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.mlp(self.mlp_norm(hidden))


class PusherBTransformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.d_vocab, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.RMSNorm(
            config.d_model,
            eps=config.normalization_epsilon,
        )
        self.unembedding = nn.Linear(
            config.d_model,
            config.d_vocab,
            bias=False,
        )
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initialization_std,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if tokens.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds configured context length")
        hidden = self.token_embedding(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.unembedding(self.final_norm(hidden))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
