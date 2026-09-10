"""Paper-scale transformer configured for binary RRXOR tokens."""

from experiments.mess3_supervised.paper_supervised_replication.model import (
    PaperModelConfig,
    PaperTransformer,
    parameter_count,
)


MODEL_CONFIG = PaperModelConfig(d_vocab=2)


class RRXORPaperTransformer(PaperTransformer):
    """Stable experiment-local module path for the supervised RRXOR model."""
