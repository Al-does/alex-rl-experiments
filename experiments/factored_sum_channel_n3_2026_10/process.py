"""N sharp Mess3 factors observed through a "sum channel".

Each step every factor emits a Mess3 sub-token (edge-emitting, alpha=0.85,
x=0.05). With probability ``1 - epsilon`` the observed token is the full
sub-token tuple (``3**N`` tokens, Cartesian order); with probability
``epsilon`` it is only the sum of the sub-tokens mod 3 (3 more tokens).
Latent dynamics stay independent, but a sum-only observation couples the
factors' beliefs (explaining away), so the exact predictive leaves the
product-state manifold. ``epsilon = 0`` is the fully factored control; the
vocabulary is kept identical (the sum tokens just never occur).

:func:`run_filter` gives the exact Bayesian filter and, with ``project=True``,
the factored assumed-density filter (re-projected onto the product of factor
marginals after every update). Its CE is the reference for what a purely
factored belief representation can achieve.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import numpy as np
from envs.hmm import HMMModel, stationary_distribution
from envs.hmm.factored import (
    compose_hmm_factors,
    factor_marginals,
    product_distribution,
)
from envs.hmm.jax_env import JaxHMMEnv

MESS3_ALPHA = 0.85
MESS3_X = 0.05
SUB_TOKENS = 3
FACTOR_STATES = 3
EPSILON = 0.5
EPISODE_LENGTH = 127
CONTEXT_LENGTH = 128


def mess3_edges(alpha: float = MESS3_ALPHA, x: float = MESS3_X) -> np.ndarray:
    """``T[k, s, t]`` of Shai et al. (2026) App. C.1.1 (token k, source s, dest t)."""

    beta, y = (1.0 - alpha) / 2.0, 1.0 - 2.0 * x
    k, s, t = np.ix_(range(SUB_TOKENS), range(FACTOR_STATES), range(FACTOR_STATES))
    return np.where(t == k, alpha, beta) * np.where(s == t, y, x)


def mess3_model(alpha: float = MESS3_ALPHA, x: float = MESS3_X) -> HMMModel:
    edges = mess3_edges(alpha, x)
    transition = edges.sum(axis=0)
    return HMMModel(
        initial_distribution=stationary_distribution(transition),
        transition_matrix=transition,
        emission_matrix=edges.sum(axis=2).T,
        edge_transition_matrices=edges,
    )


@dataclass(frozen=True)
class SumChannel:
    n_factors: int
    epsilon: float = EPSILON
    alpha: float = MESS3_ALPHA
    x: float = MESS3_X

    def __post_init__(self) -> None:
        if self.n_factors < 2:
            raise ValueError("need at least two factors")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")

    @property
    def tuple_tokens(self) -> int:
        return SUB_TOKENS**self.n_factors

    @property
    def token_count(self) -> int:
        """Emitted tokens: every sub-token tuple, then the 3 sum-only tokens."""

        return self.tuple_tokens + SUB_TOKENS

    @property
    def bos_token(self) -> int:
        return self.token_count

    @property
    def vocab_size(self) -> int:
        return self.token_count + 1

    @property
    def n_states(self) -> int:
        return FACTOR_STATES**self.n_factors

    @property
    def factor_sizes(self) -> tuple[int, ...]:
        return (FACTOR_STATES,) * self.n_factors

    @cached_property
    def model(self) -> HMMModel:
        factor = mess3_model(self.alpha, self.x)
        cartesian = compose_hmm_factors([factor] * self.n_factors)
        joint = np.asarray(cartesian.edge_transition_matrices)
        sums = np.array([sum(t) % SUB_TOKENS for t in np.ndindex(self.factor_sizes)])
        sum_edges = np.stack([joint[sums == r].sum(axis=0) for r in range(SUB_TOKENS)])
        edges = np.concatenate([(1.0 - self.epsilon) * joint, self.epsilon * sum_edges])
        return HMMModel(
            initial_distribution=cartesian.initial_distribution,
            transition_matrix=edges.sum(axis=0),
            emission_matrix=edges.sum(axis=2).T,
            edge_transition_matrices=edges,
        )

    @property
    def edges(self) -> np.ndarray:
        return np.asarray(self.model.edge_transition_matrices, dtype=np.float64)

    @property
    def initial(self) -> np.ndarray:
        return np.asarray(self.model.initial_distribution, dtype=np.float64)

    def make_env(self) -> JaxHMMEnv:
        return JaxHMMEnv.from_model(
            self.model,
            episode_length=EPISODE_LENGTH,
            reward_fn=lambda action, raw: action == raw,
            delay=1,
        )

    def marginals(self, joint: np.ndarray) -> np.ndarray:
        """``(..., S)`` joint beliefs -> ``(..., N * 3)`` concatenated factor marginals."""

        return np.concatenate(factor_marginals(joint, self.factor_sizes), axis=-1)

    def product_of_marginals(self, joint: np.ndarray) -> np.ndarray:
        return product_distribution(factor_marginals(joint, self.factor_sizes))

    def run_filter(
        self, raw: np.ndarray, *, project: bool = False, chunk: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter float64 beliefs over ``raw`` (n, L) token rows.

        Returns ``beliefs`` (n, L, S), the belief before observing ``raw[:, t]``
        (position ``t`` of the shifted CE input, which has seen BOS and
        ``raw[:, :t]``), and ``nll`` (n, L) = ``-log P(raw[:, t] | raw[:, :t])``.
        ``project=True`` is the factored assumed-density filter.
        """

        raw = np.asarray(raw)
        edges, token_given_state = self.edges, self.edges.sum(axis=-1).T
        beliefs = np.empty(raw.shape + (self.n_states,))
        nll = np.empty(raw.shape)
        for start in range(0, len(raw), chunk):
            rows = raw[start : start + chunk]
            belief = np.broadcast_to(self.initial, (len(rows), self.n_states)).copy()
            for t in range(raw.shape[1]):
                beliefs[start : start + chunk, t] = belief
                token = rows[:, t]
                predictive = np.einsum("ns,sn->n", belief, token_given_state[:, token])
                nll[start : start + chunk, t] = -np.log(predictive)
                belief = np.einsum("ns,nst->nt", belief, edges[token])
                belief /= belief.sum(axis=-1, keepdims=True)
                if project:
                    belief = self.product_of_marginals(belief)
        return beliefs, nll

    def sample_tokens(self, key: jax.Array, batch: int) -> jax.Array:
        return self.make_env().sample_tokens(key, batch, EPISODE_LENGTH)
