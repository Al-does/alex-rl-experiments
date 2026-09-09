"""Frozen four-action controlled HMM, specification version 1.0.

T[a, x, i, j] = P(X[t+1]=x, S[t+1]=j | S[t]=i, A[t]=a).
Reward is 1[S[t+1]=E]. The actor/filter observes actions and tokens only.
NumPy is the only dependency. This is an environment and exact filter, not
a trained policy or a certified Bayes-optimal POMDP planner.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

SPEC_VERSION = "1.0"
STATES = ("M1", "M2", "E", "S")
COARSE_STATES = ("M", "E", "S")
ACTIONS = ("a1", "a2", "aE", "aS")
TOKENS = (0, 1)
SPEEDS = {"half": (0.20, 0.05), "quarter": (0.10, 0.025)}
AGGREGATION = np.array([[1., 0., 0.], [1., 0., 0.],
                        [0., 1., 0.], [0., 0., 1.]])
# Rows = source, columns = destination. -1 denotes an absent fine-state edge.
EDGE_TOKENS = np.array([[0, -1, 1, -1], [-1, 0, 1, -1],
                        [-1, -1, 0, 1], [0, 1, 1, 0]], dtype=int)


def stationary_distribution(P):
    """Stationary row distribution of an irreducible stochastic matrix."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    p = np.linalg.solve(P.T - np.eye(n) + np.ones((n, n)), np.ones(n))
    p /= p.sum()
    if p.min() < -1e-12 or not np.allclose(p @ P, p):
        raise ValueError("Invalid stationary distribution")
    return p


def _belief(b, n):
    b = np.asarray(b, dtype=float)
    if b.shape != (n,) or not np.isfinite(b).all() or b.min() < 0:
        raise ValueError("Belief must be a finite nonnegative vector")
    if not np.isclose(b.sum(), 1., atol=1e-12, rtol=1e-10):
        raise ValueError("Belief probabilities must sum to one")
    return b.copy()


def _index(value, size, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value < size:
        raise ValueError(f"{name} must be between 0 and {size - 1}")
    return int(value)


@dataclass(frozen=True)
class Model:
    variant: int
    speed: str
    states: tuple
    T: np.ndarray
    reset_prior: np.ndarray

    @property
    def P(self):
        """P[action, source, destination]."""
        return self.T.sum(axis=1)

    @property
    def O(self):
        """O[action, source, token]: marginal emissions, not a sampler."""
        return self.T.sum(axis=-1).transpose(0, 2, 1)

    @property
    def G(self):
        """G[source, action]: probability of reward on the next transition."""
        return self.T[:, :, :, self.states.index("E")].sum(axis=1).T

    def predict_tokens(self, b, action):
        action = _index(action, len(ACTIONS), "action")
        return _belief(b, len(self.states)) @ self.O[action]

    def predict_rewards(self, b):
        return _belief(b, len(self.states)) @ self.G

    def update(self, b, action, token):
        """Exact posterior after action and token; reward is deliberately absent."""
        action = _index(action, len(ACTIONS), "action")
        token = _index(token, len(TOKENS), "token")
        raw = _belief(b, len(self.states)) @ self.T[action, token]
        probability = raw.sum()
        if probability <= 0:
            raise ValueError("This action/token is impossible under the supplied belief")
        return raw / probability

    def coarse(self):
        """Return the exact three-state quotient of variant 2."""
        if self.variant != 2:
            raise ValueError("Variant 3 has no exact M1/M2 quotient for all actions")
        if self.states == COARSE_STATES:
            return self
        aggregate = self.T @ AGGREGATION
        Q = aggregate[:, :, [0, 2, 3], :]
        if not np.allclose(aggregate, AGGREGATION @ Q, atol=1e-14, rtol=0):
            raise ValueError("Joint action/token quotient condition failed")
        return Model(2, self.speed, COARSE_STATES, Q.copy(), self.reset_prior @ AGGREGATION)

    def as_dict(self):
        h, l = SPEEDS[self.speed]
        return dict(spec_version=SPEC_VERSION, variant=self.variant, speed=self.speed,
            states=list(self.states), actions=list(ACTIONS), tokens=list(TOKENS),
            preferred_M_exit=h, nonpreferred_M_exit=l,
            reset_prior=self.reset_prior.tolist(),
            reset_prior_definition="stationary under uniformly random independent actions",
            reward_by_destination=[int(s == "E") for s in self.states],
            P=self.P.tolist(), O=self.O.tolist(), G=self.G.tolist(), T=self.T.tolist(),
            array_axes=dict(P=["action", "source", "destination"],
                O=["action", "source", "token"], G=["source", "action"],
                T=["action", "token", "source", "destination"]))


def build_model(variant=3, speed="half", coarse=False):
    """Construct one of the four frozen environments, or a variant-2 quotient."""
    if variant not in (2, 3):
        raise ValueError("variant must be 2 or 3")
    if speed not in SPEEDS:
        raise ValueError("speed must be 'half' or 'quarter'")
    h, l = SPEEDS[speed]
    exits1 = np.array([h, l, l, l])
    exits2 = exits1.copy() if variant == 2 else np.array([l, h, l, l])
    e = np.array([.300, .300, .490, .465])
    s_loop = np.array([.155, .155, .320, .435])
    s_m1 = np.array([.145, .145, .170, .030])
    s_m2 = np.array([.400, .400, .020, .070])
    P = np.zeros((4, 4, 4))
    P[:, 0, 0] = 1 - exits1; P[:, 0, 2] = exits1
    P[:, 1, 1] = 1 - exits2; P[:, 1, 2] = exits2
    P[:, 2, 2] = e; P[:, 2, 3] = 1 - e
    P[:, 3, 0] = s_m1; P[:, 3, 1] = s_m2
    P[:, 3, 2] = e; P[:, 3, 3] = s_loop
    T = np.stack([P * (EDGE_TOKENS == token)[None, :, :] for token in TOKENS], axis=1)
    if T.min() < 0 or not np.allclose(T.sum(axis=(1, 3)), 1):
        raise ValueError("Kernel is not stochastic")
    prior = stationary_distribution(P.mean(axis=0))
    model = Model(variant, speed, STATES, T, prior)
    return model.coarse() if coarse else model


class ControlledHMM:
    """Minimal continuing environment; no implicit termination or episode limit.

    reset() emits no HMM token and returns None. step(action) returns only
    (token, reward). Use hidden_state_for_evaluation solely for labels/debugging.
    A reset prior override must also be passed to the actor's Bayesian filter.
    """
    def __init__(self, model: Model, seed: Optional[int] = None):
        self.model = model
        self.rng = np.random.default_rng(seed)
        self._state = None

    def reset(self, prior=None):
        p = self.model.reset_prior if prior is None else _belief(prior, len(self.model.states))
        self._state = int(self.rng.choice(len(self.model.states), p=p))
        return None

    @property
    def hidden_state_for_evaluation(self):
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Call reset() first")
        action = _index(action, len(ACTIONS), "action")
        n = len(self.model.states)
        # Sample token and destination TOGETHER. This also works in the quotient,
        # where S->M has two parallel emission-labelled transitions.
        probabilities = self.model.T[action, :, self._state, :].reshape(-1)
        draw = int(self.rng.choice(len(TOKENS) * n, p=probabilities))
        token, self._state = divmod(draw, n)
        reward = float(self.model.states[self._state] == "E")
        return token, reward


if __name__ == "__main__":
    model = build_model(variant=3, speed="half")
    env = ControlledHMM(model, seed=10)
    env.reset()
    belief = model.reset_prior.copy()
    actor_rng = np.random.default_rng(11)
    for t in range(8):
        # Uniform actions illustrate the API; they are not an optimal policy.
        action = int(actor_rng.integers(4))
        token, reward = env.step(action)
        belief = model.update(belief, action, token)
        print(t + 1, ACTIONS[action], token, reward, np.round(belief, 4))
