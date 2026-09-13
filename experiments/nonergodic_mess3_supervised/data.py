"""Exact nonergodic MESS3 sequence sampling and Bayesian targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_COUNT,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    TOKEN_COUNT,
    nonergodic_mess3_model,
)


BOS_TOKEN = TOKEN_COUNT
VOCAB_SIZE = TOKEN_COUNT + 1


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    tokens: np.ndarray
    components: np.ndarray
    states: np.ndarray


class NonergodicSequenceSampler:
    """Sample complete sequences from the exact six-state composite HMM."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        model = nonergodic_mess3_model()
        self.initial_distribution = np.asarray(
            model.initial_distribution,
            dtype=np.float64,
        )
        self.edge_matrices = np.asarray(
            model.edge_transition_matrices,
            dtype=np.float64,
        )
        flattened = self.edge_matrices.transpose(1, 0, 2).reshape(
            STATE_COUNT,
            TOKEN_COUNT * STATE_COUNT,
        )
        self._edge_cdf = np.cumsum(flattened, axis=1)
        self._edge_cdf[:, -1] = 1.0

    @property
    def rng_state(self) -> dict:
        return self._rng.bit_generator.state

    @rng_state.setter
    def rng_state(self, value: dict) -> None:
        self._rng.bit_generator.state = value

    def sample(
        self,
        batch_size: int,
        *,
        emission_count: int = EPISODE_LENGTH,
    ) -> SequenceBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if emission_count <= 0:
            raise ValueError("emission_count must be positive")
        if emission_count + 1 > CONTEXT_LENGTH:
            raise ValueError("sequence exceeds the paper context length")

        states = np.empty((batch_size, emission_count + 1), dtype=np.int64)
        tokens = np.full(
            (batch_size, emission_count + 1),
            BOS_TOKEN,
            dtype=np.int64,
        )
        states[:, 0] = self._rng.choice(
            STATE_COUNT,
            size=batch_size,
            p=self.initial_distribution,
        )
        components = states[:, 0] // STATES_PER_COMPONENT

        for position in range(1, emission_count + 1):
            current = states[:, position - 1]
            uniforms = self._rng.random(batch_size)
            choices = (
                uniforms[:, None] > self._edge_cdf[current]
            ).sum(axis=1)
            tokens[:, position] = choices // STATE_COUNT
            states[:, position] = choices % STATE_COUNT

        if not np.all(
            states // STATES_PER_COMPONENT == components[:, None]
        ):
            raise AssertionError("nonergodic samples changed component")
        return SequenceBatch(
            tokens=tokens,
            components=components,
            states=states,
        )


def weighted_beliefs(tokens: np.ndarray) -> np.ndarray:
    """Return exact six-state beliefs aligned with every input token."""

    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if tokens.shape[1] > CONTEXT_LENGTH:
        raise ValueError("sequence exceeds the paper context length")
    if not np.all(tokens[:, 0] == BOS_TOKEN):
        raise ValueError("every sequence must begin with BOS")
    if tokens.shape[1] > 1 and (
        (tokens[:, 1:] < 0).any() or (tokens[:, 1:] >= TOKEN_COUNT).any()
    ):
        raise ValueError("emissions must be token indices zero through two")

    model = nonergodic_mess3_model()
    edges = np.asarray(model.edge_transition_matrices, dtype=np.float64)
    belief = np.broadcast_to(
        np.asarray(model.initial_distribution, dtype=np.float64),
        (len(tokens), STATE_COUNT),
    ).copy()
    result = np.empty((*tokens.shape, STATE_COUNT), dtype=np.float64)
    result[:, 0] = belief
    for position in range(1, tokens.shape[1]):
        operators = edges[tokens[:, position]]
        belief = np.einsum("bi,bij->bj", belief, operators)
        belief /= belief.sum(axis=1, keepdims=True)
        result[:, position] = belief
    return result


def next_token_distributions(beliefs: np.ndarray) -> np.ndarray:
    """Map six-state beliefs to exact distributions over the three emissions."""

    beliefs = np.asarray(beliefs, dtype=np.float64)
    if beliefs.shape[-1] != STATE_COUNT:
        raise ValueError("belief width must equal the six composite states")
    emission = np.asarray(
        nonergodic_mess3_model().emission_matrix,
        dtype=np.float64,
    )
    return beliefs @ emission


def component_posteriors(beliefs: np.ndarray) -> np.ndarray:
    beliefs = np.asarray(beliefs, dtype=np.float64)
    if beliefs.shape[-1] != STATE_COUNT:
        raise ValueError("belief width must equal the six composite states")
    return beliefs.reshape(
        *beliefs.shape[:-1],
        COMPONENT_COUNT,
        STATES_PER_COMPONENT,
    ).sum(axis=-1)
