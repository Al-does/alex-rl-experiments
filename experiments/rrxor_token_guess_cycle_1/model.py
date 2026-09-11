"""Paper-scale actor-critic selected for the RRXOR experiment."""

from experiments.mess3_token_guess_cycle_2.model import (
    PaperActorCriticConfig,
    PaperActorCriticModel,
)


class RRXORActorCritic(PaperActorCriticModel):
    """Stable experiment-local RLlib module path."""


MODEL_CONFIG = PaperActorCriticConfig(
    context_length=10,
    max_seq_len=32,
).to_dict()
