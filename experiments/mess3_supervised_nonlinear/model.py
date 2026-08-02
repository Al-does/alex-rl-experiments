"""Paper-faithful MESS3 transformer with a nonlinear prediction head."""

from __future__ import annotations

from torch import nn

from experiments.mess3_supervised.paper_supervised_replication.model import (
    PaperModelConfig,
    PaperTransformer,
)


class SwishMLPDecoderTransformer(PaperTransformer):
    """Change only the paper model's token-prediction decoder."""

    def __init__(
        self,
        config: PaperModelConfig | None = None,
        *,
        decoder_depth: int,
        decoder_hidden_dim: int = 64,
    ):
        if decoder_depth < 2:
            raise ValueError("the nonlinear decoder must have at least 2 layers")
        if decoder_hidden_dim <= 0:
            raise ValueError("decoder_hidden_dim must be positive")

        # Constructing the complete paper model first preserves identical
        # transformer-backbone initialization for a fixed seed. Only the
        # original linear unembedding is then replaced.
        super().__init__(config)
        self.decoder_depth = decoder_depth
        self.decoder_hidden_dim = decoder_hidden_dim

        layers: list[nn.Module] = []
        input_dim = self.config.d_model
        for layer_index in range(decoder_depth):
            is_output = layer_index == decoder_depth - 1
            output_dim = (
                self.config.d_vocab if is_output else decoder_hidden_dim
            )
            layers.append(nn.Linear(input_dim, output_dim, bias=True))
            if not is_output:
                # PyTorch's SiLU is the standard Swish activation x * sigmoid(x).
                layers.append(nn.SiLU())
            input_dim = output_dim

        self.unembedding = nn.Sequential(*layers)
        self.unembedding.apply(self._initialize)

    @property
    def prediction_head_spec(self) -> dict[str, object]:
        return {
            "type": "mlp",
            "depth": self.decoder_depth,
            "hidden_dim": self.decoder_hidden_dim,
            "activation": "swish_silu",
            "activation_after_output": False,
        }
