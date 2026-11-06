"""Full-State Quotient exact-filter reference (ported from scratch/fsq_pred.py).

Nearest-grid discounted value iteration is a feasible near-optimal reference,
not an optimality certificate. Filtering always follows the executed actions.
"""

from __future__ import annotations

import numpy as np

LAMBDAS = (0.0, 0.25, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.9, 1.0)
BASE = np.array([[.75, .15, .10], [.15, .75, .10], [.30, .30, .40]])
EMISSION = np.full((3, 3), .075)
np.fill_diagonal(EMISSION, .85)
DIRECTIONS = np.array([[0., 0., 0.], [1., -1., -1.], [-1., 1., -1.]])
TRANSITIONS = np.repeat(BASE[None], 3, axis=0)
TRANSITIONS[:, :, 2] *= np.exp(1.5 * DIRECTIONS)
TRANSITIONS /= TRANSITIONS.sum(axis=2, keepdims=True)


def target(belief: np.ndarray, mode: str) -> np.ndarray:
    b = np.asarray(belief, dtype=np.float64)
    if b.shape[-1] != 3 or not np.isfinite(b).all():
        raise ValueError("belief must have three finite coordinates")
    if mode == "mirror":
        return b[..., [1, 0, 2]].copy()
    if mode != "point":
        raise ValueError(f"unknown intervention: {mode}")
    out = np.zeros_like(b)
    out[..., 2] = b[..., 2]
    lower = b[..., 0] < b[..., 1]
    out[..., 0] = np.where(lower, 1 - b[..., 2], 0)
    out[..., 1] = np.where(lower, 0, 1 - b[..., 2])
    return out


class BeliefPolicy:
    def __init__(self, n: int = 60, gamma: float = .97, iterations: int = 800):
        if n < 1 or not 0 < gamma < 1 or iterations < 1:
            raise ValueError("invalid value-iteration settings")
        self.n = n
        self.gamma = gamma
        self.iterations = iterations
        self.grid = np.array([(i/n, j/n, 1-(i+j)/n)
                              for i in range(n+1) for j in range(n+1-i)])
        self.lookup = np.full((n+1, n+1), -1, dtype=int)
        ij = np.rint(self.grid[:, :2] * n).astype(int)
        self.lookup[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
        prior = np.einsum("ks,asj->kaj", self.grid, TRANSITIONS)
        weights = prior[:, :, None, :] * EMISSION.T[None, None, :, :]
        probability = weights.sum(axis=-1)
        nxt = self.nearest(weights / probability[..., None])
        value = np.zeros(len(self.grid))
        for _ in range(iterations):
            q = self.grid[:, 2, None] + gamma * (probability * value[nxt]).sum(-1)
            value = q.max(axis=1)
        self.actions = q.argmax(axis=1)

    def nearest(self, belief: np.ndarray) -> np.ndarray:
        b = np.asarray(belief)
        ij = np.rint(b[..., :2] * self.n).astype(int)
        i, j = ij[..., 0], ij[..., 1]
        over = i+j > self.n
        i, j = (np.where(over & (b[..., 0] > b[..., 1]), self.n-j, i),
                np.where(over & (b[..., 0] <= b[..., 1]), self.n-i, j))
        return self.lookup[i, j]

    def __call__(self, belief: np.ndarray) -> np.ndarray:
        return self.actions[self.nearest(belief)]


def simulate(policy: BeliefPolicy, mode: str, strength: float, *,
             episodes: int = 32, horizon: int = 1024, seed: int = 2042,
             random_actions: bool = False) -> dict:
    if episodes < 2 or horizon < 1 or not 0 <= strength <= 1:
        raise ValueError("invalid simulation budget or strength")
    rng = np.random.default_rng(seed)
    state = rng.integers(3, size=episodes)
    prior = np.full((episodes, 3), 1/3)
    occupied = np.zeros(episodes)
    counts = np.zeros(3, dtype=int)
    for _ in range(horizon):
        u = rng.random(episodes)
        obs = (u[:, None] >= EMISSION[state].cumsum(axis=1)).sum(axis=1)
        b = prior * EMISSION[:, obs].T
        b /= b.sum(axis=1, keepdims=True)
        occupied += state == 2
        installed = (1-strength)*b + strength*target(b, mode)
        action = rng.integers(3, size=episodes) if random_actions else policy(installed)
        counts += np.bincount(action, minlength=3)
        probabilities = TRANSITIONS[action, state]
        state = (rng.random(episodes)[:, None] >= probabilities.cumsum(axis=1)).sum(1)
        prior = np.einsum("bs,bsj->bj", b, TRANSITIONS[action])
    occupied /= horizon
    return {"occupancy": float(occupied.mean()),
            "occupancy_se": float(occupied.std(ddof=1)/np.sqrt(episodes)),
            "action_fractions": (counts/(episodes*horizon)).tolist()}


def prediction_curve(policy: BeliefPolicy, **kwargs) -> dict:
    return {mode: {str(l): simulate(policy, mode, l, **kwargs) for l in LAMBDAS}
            for mode in ("point", "mirror")}
