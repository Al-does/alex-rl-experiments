"""Antithetic OpenAI-ES with centered-rank shaping and Adam (Salimans et al. 2017)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def centered_ranks(values: np.ndarray) -> np.ndarray:
    """Map fitness values to ranks in ``[-0.5, 0.5]``; ties share their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.zeros(1)
    order = values.argsort(kind="stable")
    ranks = np.empty(values.size)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks / (values.size - 1) - 0.5


@dataclass(slots=True)
class ESState:
    """Serializable outer-loop state."""

    mean: np.ndarray
    generation: int = 0
    adam_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    adam_v: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64)
        if self.adam_m.size == 0:
            self.adam_m = np.zeros_like(self.mean)
        if self.adam_v.size == 0:
            self.adam_v = np.zeros_like(self.mean)

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "mean": self.mean.tolist(),
            "adam_m": self.adam_m.tolist(),
            "adam_v": self.adam_v.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ESState:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            generation=int(payload["generation"]),
            adam_m=np.asarray(payload["adam_m"], dtype=np.float64),
            adam_v=np.asarray(payload["adam_v"], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class OpenES:
    """Fitness-ascent ES hyperparameters."""

    num_pairs: int
    sigma: float
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0

    def noise(self, state: ESState, seed: int) -> np.ndarray:
        """``(num_pairs, dim)`` perturbations, reproducible per generation."""
        rng = np.random.default_rng([seed, state.generation])
        return rng.standard_normal((self.num_pairs, state.mean.size))

    def candidates(self, state: ESState, noise: np.ndarray) -> np.ndarray:
        """Rows ``[mu + s*e_0, mu - s*e_0, mu + s*e_1, ...]``."""
        plus = state.mean + self.sigma * noise
        minus = state.mean - self.sigma * noise
        return np.stack([plus, minus], axis=1).reshape(-1, state.mean.size)

    def gradient(self, noise: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Antithetic ES estimate of the ascent direction from interleaved fitness."""
        shaped = centered_ranks(np.asarray(fitness)).reshape(self.num_pairs, 2)
        weights = shaped[:, 0] - shaped[:, 1]
        return weights @ noise / (2 * self.num_pairs * self.sigma)

    def step(self, state: ESState, noise: np.ndarray, fitness: np.ndarray) -> ESState:
        grad = self.gradient(noise, fitness) - self.weight_decay * state.mean
        t = state.generation + 1
        m = self.beta1 * state.adam_m + (1 - self.beta1) * grad
        v = self.beta2 * state.adam_v + (1 - self.beta2) * grad**2
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)
        mean = state.mean + self.learning_rate * m_hat / (
            np.sqrt(v_hat) + self.adam_eps
        )
        return ESState(mean=mean, generation=t, adam_m=m, adam_v=v)
