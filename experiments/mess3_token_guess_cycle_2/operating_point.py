"""Choosing MESS3 transition and emission parameters for metric sensitivity.

`REVIEW.md` found that at the parameters cycle 1 used (`alpha=0.85`, self-transition
0.9) an affine probe on the raw observations already scores R² = 0.967, leaving
the belief-probe metric only 0.03 to move through. The natural fix is to change
the process so that the belief is a less linear function of the observation
history.

Two things have to be checked before committing to a new operating point, and
they pull in opposite directions.

Lowering the floor means slowing the chain, because a fast-mixing chain lets the
belief be approximated by an exponentially weighted average of recent one-hot
observations, which is exactly what an affine probe computes. But slowing the
chain also raises the autocorrelation of any rollout drawn from it, and the probe
is estimated from a rollout. Past some point the metric gains range and loses
precision at the same rate, and nothing is won.

This module measures both, plus the context length the belief actually requires,
so the operating point can be chosen rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.mess3_token_guess_cycle_1.analysis import PROBE_RANK
from experiments.mess3_token_guess_cycle_2.metric_references import (
    _probe_r2,
    TokenStream,
)

N_CHAINS = 4_000
# `collect_probe_data` steps this many environments in parallel, so a probe
# rollout is this many trajectories rather than that many independent draws.
PROBE_CHAINS = 16
BURN_IN = 600
FLOOR_WINDOWS = (4, 8, 16, 32, 64)
CONTEXT_LENGTHS = (8, 16, 32, 64, 128)
BLOCK = 512
RESAMPLES = 300


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """One symmetric MESS3 process, described by its two free parameters."""

    alpha: float
    self_transition: float

    @property
    def transition_matrix(self) -> np.ndarray:
        matrix = np.full((3, 3), (1.0 - self.self_transition) / 2.0)
        np.fill_diagonal(matrix, self.self_transition)
        return matrix

    @property
    def emission_matrix(self) -> np.ndarray:
        matrix = np.full((3, 3), (1.0 - self.alpha) / 2.0)
        np.fill_diagonal(matrix, self.alpha)
        return matrix

    @property
    def state_correlation_time(self) -> float:
        """1 / (1 - second eigenvalue), the timescale the hidden state persists."""

        return 1.0 / (1.0 - (3.0 * self.self_transition - 1.0) / 2.0)


def simulate_parallel(
    point: OperatingPoint,
    *,
    n_steps: int,
    seed: int,
    window: int = max(FLOOR_WINDOWS),
    n_chains: int = N_CHAINS,
) -> TokenStream:
    """Draw from ``n_chains`` chains stepped together, after burning each in.

    The chain count controls how correlated the returned samples are, which is a
    property of the collection scheme rather than of the process. Use the default
    for estimating a population quantity like the floor, and ``PROBE_CHAINS`` when
    estimating how precise a real probe rollout would be.
    """

    transition = point.transition_matrix
    emission = point.emission_matrix
    stationary = np.full(3, 1.0 / 3.0)
    rng = np.random.default_rng(seed)
    cumulative_emission = emission.cumsum(1)
    cumulative_transition = transition.cumsum(1)

    state = rng.choice(3, size=n_chains, p=stationary)
    belief = np.repeat(stationary[None, :], n_chains, axis=0)
    history = np.zeros((n_chains, window), dtype=np.int64)

    beliefs, tokens, windows = [], [], []
    per_chain = int(np.ceil(n_steps / n_chains))
    for step in range(BURN_IN + per_chain):
        token = (rng.random(n_chains)[:, None] > cumulative_emission[state]).sum(1)
        if step >= BURN_IN:
            beliefs.append(belief.copy())
            tokens.append(token.copy())
            windows.append(history.copy())
        history = np.concatenate([history[:, 1:], token[:, None]], axis=1)
        posterior = belief * emission[:, token].T
        belief = (posterior / posterior.sum(1, keepdims=True)) @ transition
        state = (rng.random(n_chains)[:, None] > cumulative_transition[state]).sum(1)
    # Order by chain so that contiguous slices are contiguous in time, which the
    # block bootstrap relies on.
    return TokenStream(
        beliefs=np.concatenate(beliefs).reshape(-1, n_chains, 3).transpose(1, 0, 2).reshape(-1, 3)[:n_steps],
        tokens=np.concatenate(tokens).reshape(-1, n_chains).T.reshape(-1)[:n_steps],
        windows=np.concatenate(windows).reshape(-1, n_chains, window).transpose(1, 0, 2).reshape(-1, window)[:n_steps],
    )


def _one_hot(windows: np.ndarray, k: int) -> np.ndarray:
    return np.eye(3, dtype=np.float64)[windows[:, -k:]].reshape(len(windows), -1)


def belief_r2_floor(fit: TokenStream, test: TokenStream) -> float:
    """The best affine readout of the raw observations, over window lengths."""

    return max(
        float(_probe_r2(_one_hot(fit.windows, k), fit.beliefs, _one_hot(test.windows, k), test.beliefs)[0])
        for k in FLOOR_WINDOWS
    )


def accuracy_bounds(point: OperatingPoint, test: TokenStream) -> tuple[float, float]:
    """Bayes-optimal accuracy, and the repeat-the-previous-observation rule."""

    emission = point.emission_matrix
    ceiling = float((( test.beliefs @ emission).argmax(1) == test.tokens).mean())
    belief = np.repeat(np.full(3, 1.0 / 3.0)[None, :], len(test.windows), axis=0)
    belief = belief * emission[:, test.windows[:, -1]].T
    belief /= belief.sum(1, keepdims=True)
    belief = belief @ point.transition_matrix
    floor = float(((belief @ emission).argmax(1) == test.tokens).mean())
    return floor, ceiling


def context_requirement(
    point: OperatingPoint,
    test: TokenStream,
    *,
    context_lengths: tuple[int, ...] = CONTEXT_LENGTHS,
) -> dict[int, float]:
    """Best belief R² available to a model limited to ``k`` observations.

    The exact posterior given the last ``k`` observations under a stationary
    prior is what a ``k``-context model can compute at best, so this is an
    architecture ceiling that no training objective can exceed.
    """

    transition = point.transition_matrix
    emission = point.emission_matrix
    scores: dict[int, float] = {}
    for k in context_lengths:
        if k > test.windows.shape[1]:
            continue
        belief = np.repeat(np.full(3, 1.0 / 3.0)[None, :], len(test.windows), axis=0)
        for offset in range(k):
            belief = belief * emission[:, test.windows[:, -k + offset]].T
            belief /= belief.sum(1, keepdims=True)
            belief = belief @ transition
        residual = ((belief - test.beliefs) ** 2).sum()
        total = ((test.beliefs - test.beliefs.mean(0)) ** 2).sum()
        scores[k] = float(1.0 - residual / total)
    return scores


def integrated_autocorrelation(point: OperatingPoint, *, n_steps: int, seed: int) -> float:
    """Sum the belief autocorrelation along a single chain.

    Rollouts are drawn as trajectories, so this is what divides the nominal probe
    size to give the number of independent samples it really contains.
    """

    transition = point.transition_matrix
    emission = point.emission_matrix
    cumulative_emission = emission.cumsum(1)
    cumulative_transition = transition.cumsum(1)
    rng = np.random.default_rng(seed)
    stationary = np.full(3, 1.0 / 3.0)
    state = int(rng.choice(3, p=stationary))
    belief = stationary.copy()
    series = np.empty(n_steps)
    for step in range(n_steps + BURN_IN):
        token = int((rng.random() > cumulative_emission[state]).sum())
        if step >= BURN_IN:
            series[step - BURN_IN] = belief[0]
        posterior = belief * emission[:, token]
        belief = (posterior / posterior.sum()) @ transition
        state = int((rng.random() > cumulative_transition[state]).sum())

    centred = series - series.mean()
    variance = centred.var()
    tau = 1.0
    for lag in range(1, min(4_000, n_steps // 4)):
        value = float((centred[:-lag] * centred[lag:]).mean() / variance)
        if value <= 0.0:
            break
        tau += 2.0 * value
    return tau


def block_bootstrap_interval(
    fit: TokenStream,
    test: TokenStream,
    *,
    window: int,
    seed: int,
    block: int = BLOCK,
    resamples: int = RESAMPLES,
) -> tuple[float, float]:
    """Resample contiguous blocks, which an i.i.d. bootstrap would understate."""

    _, predicted = _probe_r2(
        _one_hot(fit.windows, window),
        fit.beliefs,
        _one_hot(test.windows, window),
        test.beliefs,
    )
    rng = np.random.default_rng(seed)
    n = len(test.beliefs)
    n_blocks = max(1, n // block)
    scores = []
    for _ in range(resamples):
        starts = rng.integers(0, max(1, n - block), n_blocks)
        index = (starts[:, None] + np.arange(block)[None, :]).ravel()
        index = index[index < n]
        residual = ((predicted[index] - test.beliefs[index]) ** 2).sum()
        total = (
            (test.beliefs[index] - test.beliefs[index].mean(0)) ** 2
        ).sum()
        scores.append(1.0 - residual / total)
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def evaluate(
    point: OperatingPoint,
    *,
    fit_steps: int,
    test_steps: int,
    seed: int,
) -> dict[str, float | dict[int, float]]:
    """Score one candidate operating point on range and on precision."""

    fit = simulate_parallel(point, n_steps=fit_steps, seed=seed)
    test = simulate_parallel(point, n_steps=test_steps, seed=seed + 1)
    floor = belief_r2_floor(fit, test)
    accuracy_floor, accuracy_ceiling = accuracy_bounds(point, test)
    tau = integrated_autocorrelation(point, n_steps=min(120_000, 40 * test_steps), seed=seed + 2)
    # Estimate the probe's precision from a rollout collected the way the study
    # collects one, not from the near-independent sample used above.
    probe_fit = simulate_parallel(
        point, n_steps=fit_steps, seed=seed + 4, n_chains=PROBE_CHAINS
    )
    probe_test = simulate_parallel(
        point, n_steps=test_steps, seed=seed + 5, n_chains=PROBE_CHAINS
    )
    low, high = block_bootstrap_interval(probe_fit, probe_test, window=32, seed=seed + 3)
    return {
        "alpha": point.alpha,
        "self_transition": point.self_transition,
        "state_correlation_time": point.state_correlation_time,
        "belief_r2_floor": floor,
        "belief_r2_range": 1.0 - floor,
        "accuracy_floor": accuracy_floor,
        "accuracy_ceiling": accuracy_ceiling,
        "accuracy_range": accuracy_ceiling - accuracy_floor,
        "context_requirement": context_requirement(point, test),
        "integrated_autocorrelation": tau,
        "effective_sample_size": test_steps / tau,
        "probe_block_bootstrap_ci": [low, high],
        "probe_ci_half_width": (high - low) / 2.0,
        "probe_rank": PROBE_RANK,
    }
