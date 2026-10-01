"""Learned Policy Optimisation (LPO) drift objective in JAX.

Mirror-learning view of PPO (Kuba et al. 2022; Lu et al. 2022): the clipped
surrogate is ``r * A - F(r, A)`` with PPO's drift
``F_ppo = relu((r - clip(r, 1 - eps, 1 + eps)) * A)``. LPO replaces ``F`` with
a small network whose parameters are meta-learned in an outer loop:

    F(r, A) = relu(w_ppo * F_ppo(r, A) + g(x(r, A)) - g(x(1, A)))

``g`` is a tanh MLP over ratio/advantage features. Its output layer starts at
zero and ``w_ppo`` at one, so the initial meta-parameters are exactly PPO.
Subtracting ``g`` at ``r = 1`` keeps ``F(1, A) = 0``; the relu keeps ``F >= 0``.

Flat meta-parameter vector: ``[g.W1, g.b1, g.W2, g.b2, w_ppo, log_entropy_coeff]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

NUM_FEATURES = 8
LOG_RATIO_BOUND = 5.0


@dataclass(frozen=True)
class DriftSpec:
    hidden: int = 32
    clip_param: float = 0.2
    initial_entropy_coeff: float = 0.05

    @property
    def num_params(self) -> int:
        return NUM_FEATURES * self.hidden + self.hidden + self.hidden + 1 + 2

    def initial_params(self, seed: int) -> np.ndarray:
        """PPO-equivalent meta-parameters with a seeded hidden layer."""
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, 1.0 / math.sqrt(NUM_FEATURES), (self.hidden, NUM_FEATURES))
        b1 = np.zeros(self.hidden)
        w2 = np.zeros(self.hidden)
        b2 = np.zeros(1)
        extra = np.array([1.0, math.log(self.initial_entropy_coeff)])
        return np.concatenate([w1.ravel(), b1, w2, b2, extra]).astype(np.float64)


def drift_features(ratio: jax.Array, advantages: jax.Array) -> jax.Array:
    """LPO-style inputs: ratio deviations and their products with ``A``."""
    log_ratio = jnp.clip(
        jnp.log(jnp.maximum(ratio, 1e-8)), -LOG_RATIO_BOUND, LOG_RATIO_BOUND
    )
    base = jnp.stack(
        [ratio - 1.0, log_ratio, (ratio - 1.0) ** 2, log_ratio**2], axis=-1
    )
    return jnp.concatenate([base, base * advantages[..., None]], axis=-1)


def _unpack(spec: DriftSpec, meta: jax.Array):
    h, f = spec.hidden, NUM_FEATURES
    sizes = [f * h, h, h, 1, 1, 1]
    offsets = np.cumsum([0, *sizes])
    w1, b1, w2, b2, w_ppo, log_entropy = (
        meta[offsets[i] : offsets[i + 1]] for i in range(len(sizes))
    )
    return w1.reshape(h, f), b1, w2, b2[0], w_ppo[0], log_entropy[0]


def ppo_drift(spec: DriftSpec, ratio: jax.Array, advantages: jax.Array) -> jax.Array:
    eps = spec.clip_param
    return jax.nn.relu((ratio - jnp.clip(ratio, 1.0 - eps, 1.0 + eps)) * advantages)


def drift(
    spec: DriftSpec, meta: jax.Array, ratio: jax.Array, advantages: jax.Array
) -> jax.Array:
    w1, b1, w2, b2, w_ppo, _ = _unpack(spec, meta)

    def g(features):
        return jnp.tanh(features @ w1.T + b1) @ w2 + b2

    learned = g(drift_features(ratio, advantages)) - g(
        drift_features(jnp.ones_like(ratio), advantages)
    )
    return jax.nn.relu(w_ppo * ppo_drift(spec, ratio, advantages) + learned)


def entropy_coeff(spec: DriftSpec, meta: jax.Array) -> jax.Array:
    return jnp.exp(_unpack(spec, meta)[-1])


def objective(spec: DriftSpec):
    """``ppo.Objective``: surrogate ``r * A - F_meta(r, A)``, learned entropy."""

    def fn(meta, ratio, advantages):
        surrogate = ratio * advantages - drift(spec, meta, ratio, advantages)
        return surrogate, entropy_coeff(spec, meta)

    return fn
