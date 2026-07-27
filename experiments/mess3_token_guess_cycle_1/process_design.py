"""Score a candidate HMM as a measuring instrument for belief-geometry work.

The token-guess study reports two numbers: greedy accuracy and belief-probe R².
Whether either can separate training objectives is a property of the *process*,
fixed before a single step is trained. This module makes that property
measurable so an operating point can be chosen deliberately.

The two quantities that matter are headroom, not level:

* how far the Bayes-optimal policy sits above the best policy that only looks
  at the last few tokens, and
* how much belief variance sits between what those same tokens already explain
  and what the task's own sufficient statistic can explain.

The upper end of the second range is easy to overlook. A policy objective can
only reward what the one-step predictive distribution distinguishes, so if the
token alphabet is smaller than the state space that distribution is a strict
projection of the belief and no amount of training can drive the probe to one.
Alphabet size is therefore a ceiling, not just a difficulty knob.

Everything here is exact given the belief trajectory. No agent is trained, so
a candidate process can be rejected before it costs any GPU time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LAG_DEPTHS = (1, 2, 3, 8)
PROBE_RANK = 2


@dataclass(frozen=True, slots=True)
class BeliefTrajectory:
    """Delay-one decision stream, one row per independent chain."""

    beliefs: np.ndarray
    tokens: np.ndarray
    predictive: np.ndarray

    @property
    def n_chains(self) -> int:
        return self.beliefs.shape[0]


def spectral_gap(transition: np.ndarray) -> float:
    """Second-largest eigenvalue modulus: how fast the chain forgets."""

    eigenvalues = np.linalg.eigvals(np.asarray(transition, dtype=np.float64))
    return float(np.sort(np.abs(eigenvalues))[-2])


def stationary(transition: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(np.asarray(transition, dtype=np.float64).T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return vector / vector.sum()


def simulate(
    transition: np.ndarray,
    emission: np.ndarray,
    *,
    n_chains: int = 512,
    n_steps: int = 420,
    burn_in: int = 96,
    seed: int = 0,
) -> BeliefTrajectory:
    """Run independent chains, tracking the exact predictive belief.

    Timing matches ``HMMEnv(delay=1)``: at decision ``t`` the agent has seen
    ``x_0..x_{t-1}`` and must guess ``x_t``, so the decision-time belief is
    ``P(s_t | x_<t)``. ``burn_in`` discards the transient after the reset to
    the stationary prior, mirroring the probe's per-episode warmup.
    """

    transition = np.asarray(transition, dtype=np.float64)
    emission = np.asarray(emission, dtype=np.float64)
    n_states, n_tokens = emission.shape
    rng = np.random.default_rng(seed)

    initial = stationary(transition)
    states = rng.choice(n_states, size=n_chains, p=initial)
    beliefs = np.repeat(initial[None, :], n_chains, axis=0)

    transition_cdf = np.cumsum(transition, axis=1)
    emission_cdf = np.cumsum(emission, axis=1)
    emission_columns = emission.T

    kept_beliefs = np.empty((n_chains, n_steps, n_states))
    kept_tokens = np.empty((n_chains, n_steps), dtype=np.int64)
    for step in range(burn_in + n_steps):
        tokens = (
            rng.random((n_chains, 1)) > emission_cdf[states]
        ).sum(axis=1).clip(max=n_tokens - 1)
        if step >= burn_in:
            kept_beliefs[:, step - burn_in] = beliefs
            kept_tokens[:, step - burn_in] = tokens

        posterior = beliefs * emission_columns[tokens]
        beliefs = (posterior / posterior.sum(axis=1, keepdims=True)) @ transition
        states = (
            rng.random((n_chains, 1)) > transition_cdf[states]
        ).sum(axis=1).clip(max=n_states - 1)

    return BeliefTrajectory(
        beliefs=kept_beliefs,
        tokens=kept_tokens,
        predictive=kept_beliefs @ emission,
    )


def _windows(tokens: np.ndarray, depth: int, n_tokens: int) -> np.ndarray:
    """One-hot encode the ``depth`` tokens visible at each decision."""

    n_chains, n_steps = tokens.shape
    features = np.zeros((n_chains, n_steps, depth * n_tokens))
    for lag in range(1, depth + 1):
        chains = np.repeat(np.arange(n_chains), n_steps - lag)
        steps = np.tile(np.arange(lag, n_steps), n_chains)
        observed = tokens[chains, steps - lag]
        features[chains, steps, (lag - 1) * n_tokens + observed] = 1.0
    return features


def _window_keys(tokens: np.ndarray, depth: int, n_tokens: int) -> np.ndarray:
    """Integer label for the visible window, with a bucket for short history."""

    n_chains, n_steps = tokens.shape
    keys = np.zeros((n_chains, n_steps), dtype=np.int64)
    for lag in range(1, depth + 1):
        shifted = np.full((n_chains, n_steps), n_tokens, dtype=np.int64)
        shifted[:, lag:] = tokens[:, : n_steps - lag]
        keys = keys * (n_tokens + 1) + shifted
    return keys


def best_windowed_accuracy(
    trajectory: BeliefTrajectory,
    depth: int,
    n_tokens: int,
) -> float:
    """Accuracy of the best policy measurable from the last ``depth`` tokens.

    The policy picks ``argmax_a E[P(x_t = a) | window]``, so this is the
    ceiling for any agent that keeps no state beyond a short token window --
    including the echo heuristic, which is one such policy.
    """

    keys = _window_keys(trajectory.tokens, depth, n_tokens).ravel()
    predictive = trajectory.predictive.reshape(-1, n_tokens)
    order = np.argsort(keys, kind="stable")
    keys, predictive = keys[order], predictive[order]
    boundaries = np.flatnonzero(np.diff(keys)) + 1
    total = 0.0
    for group in np.split(predictive, boundaries):
        total += group[:, group.mean(axis=0).argmax()].sum()
    return float(total / len(predictive))


def _rank_limited_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    *,
    rank: int,
) -> float:
    feature_mean = train_features.mean(axis=0)
    target_mean = train_targets.mean(axis=0)
    centered = train_features - feature_mean
    weight, *_ = np.linalg.lstsq(centered, train_targets - target_mean, rcond=None)
    _, _, right = np.linalg.svd(centered @ weight, full_matrices=False)
    subspace = right[: min(rank, right.shape[0])].T
    weight = weight @ subspace @ subspace.T
    predicted = (test_features - feature_mean) @ weight + target_mean
    residual = float(np.square(predicted - test_targets).sum())
    total = float(np.square(test_targets - test_targets.mean(axis=0)).sum())
    return float("nan") if total == 0.0 else 1.0 - residual / total


def _split(array: np.ndarray, n_chains: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit and evaluate on disjoint chains so no history is shared."""

    half = n_chains // 2
    flat = array.reshape(n_chains, -1, array.shape[-1])
    return (
        flat[:half].reshape(-1, array.shape[-1]),
        flat[half:].reshape(-1, array.shape[-1]),
    )


def evaluate(
    transition: np.ndarray,
    emission: np.ndarray,
    *,
    name: str = "",
    n_test: int = 30_000,
    **simulate_kwargs: Any,
) -> dict[str, Any]:
    """Score one candidate process on both outcome axes."""

    emission = np.asarray(emission, dtype=np.float64)
    n_states, n_tokens = emission.shape
    trajectory = simulate(transition, emission, **simulate_kwargs)
    chains = trajectory.n_chains
    predictive = trajectory.predictive
    flat_predictive = predictive.reshape(-1, n_tokens)

    bayes = float(flat_predictive.max(axis=1).mean())
    windowed = {
        depth: best_windowed_accuracy(trajectory, depth, n_tokens)
        for depth in (1, 2, 3)
    }
    constant = float(flat_predictive.mean(axis=0).max())
    standard_error = float(np.sqrt(bayes * (1.0 - bayes) / n_test))

    rank = min(PROBE_RANK, n_states - 1)
    train_beliefs, test_beliefs = _split(trajectory.beliefs, chains)
    probes = {}
    for depth in LAG_DEPTHS:
        features = _windows(trajectory.tokens, depth, n_tokens)
        train_features, test_features = _split(features, chains)
        probes[depth] = _rank_limited_probe(
            train_features, train_beliefs, test_features, test_beliefs, rank=rank
        )
    cells = np.zeros((*predictive.shape[:2], n_tokens))
    np.put_along_axis(cells, predictive.argmax(axis=2)[..., None], 1.0, axis=2)
    train_cells, test_cells = _split(cells, chains)
    cell_probe = _rank_limited_probe(
        train_cells, train_beliefs, test_cells, test_beliefs, rank=rank
    )
    train_predictive, test_predictive = _split(predictive, chains)
    sufficient_probe = _rank_limited_probe(
        train_predictive,
        train_beliefs,
        test_predictive,
        test_beliefs,
        rank=rank,
    )

    covariance = np.cov(trajectory.beliefs.reshape(-1, n_states), rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    readout_floor = max(probes[max(LAG_DEPTHS)], cell_probe)
    return {
        "name": name,
        "n_states": n_states,
        "n_tokens": n_tokens,
        "slem": spectral_gap(transition),
        "accuracy_bayes": bayes,
        "accuracy_window1": windowed[1],
        "accuracy_window2": windowed[2],
        "accuracy_window3": windowed[3],
        "accuracy_constant": constant,
        "accuracy_headroom": bayes - windowed[1],
        "accuracy_headroom_sigma": (bayes - windowed[1]) / standard_error,
        "accuracy_headroom_window3": bayes - windowed[3],
        **{f"probe_r2_window{depth}": probes[depth] for depth in LAG_DEPTHS},
        "probe_r2_argmax_cell": cell_probe,
        "probe_r2_sufficient": sufficient_probe,
        "probe_floor": readout_floor,
        "probe_band": sufficient_probe - readout_floor,
        "belief_conditioning": float(eigenvalues[1] / eigenvalues[0]),
        "belief_max_mean": float(trajectory.beliefs.max(axis=2).mean()),
    }
