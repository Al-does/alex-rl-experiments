"""Paper-style decoder-only transformer for joint-token prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .process import joint_token_count


@dataclass(frozen=True, slots=True)
class NextTokenModelConfig:
    """PR #59's 64-wide adaptation of the paper's canonical transformer."""

    factor_count: int
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 256
    context_length: int = 9
    activation: str = "relu"
    normalization: str = "layer_norm"
    positional_embedding: str = "learned_absolute"

    def __post_init__(self) -> None:
        joint_token_count(self.factor_count)
        if min(
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_mlp,
            self.context_length,
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

    @property
    def base_vocab_size(self) -> int:
        return joint_token_count(self.factor_count)

    @property
    def bos_token(self) -> int:
        return self.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + 1

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "base_vocab_size": self.base_vocab_size,
            "bos_token": self.bos_token,
            "vocab_size": self.vocab_size,
        }


class MultiHeadCausalAttention(nn.Module):
    """Dense causal attention using PyTorch's accelerator-native SDPA."""

    def __init__(self, config: NextTokenModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
            dropout_p=0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm attention and ReLU MLP residual block."""

    def __init__(self, config: NextTokenModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadCausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.ReLU(),
            nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.mlp(self.mlp_norm(hidden))


class FactoredNextTokenTransformer(nn.Module):
    """Decoder-only LM exposing the final pre-LN residual stream."""

    def __init__(self, config: NextTokenModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        # TransformerLens omits the unembedding bias by default.
        self.unembedding = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
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

    def token_embedding_matrix(self) -> torch.Tensor:
        """Return visible joint-token embeddings, excluding BOS."""

        return self.token_embedding.weight[: self.config.base_vocab_size]

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if tokens.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds configured context length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        pre_final_norm = hidden
        logits = self.unembedding(self.final_norm(hidden))
        if return_activations:
            return logits, pre_final_norm
        return logits
