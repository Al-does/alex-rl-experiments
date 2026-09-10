from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_1.analysis import (
    N_ENVS,
    WARMUP,
    collect_probe_data,
)
from experiments.strata_token_guess_cycle_1.process import EPISODE_LENGTH


@dataclass(frozen=True, slots=True)
class GeometryData:
    activations: np.ndarray
    beliefs: np.ndarray
    observations: np.ndarray
    episode_ids: np.ndarray
    episode_steps: np.ndarray
    mask: np.ndarray


def collect_geometry_data(
    module: Any,
    *,
    n_steps: int,
    seed: np.random.SeedSequence,
    device: torch.device,
) -> GeometryData:
    if (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, (int, np.integer))
        or n_steps <= 0
    ):
        raise ValueError("n_steps must be a strictly positive integer")
    numerator = (int(n_steps) + 2 * N_ENVS * WARMUP) * EPISODE_LENGTH
    denominator = N_ENVS * (EPISODE_LENGTH - WARMUP)
    raw_budget = N_ENVS * ((numerator + denominator - 1) // denominator)
    while True:
        raw = collect_probe_data(
            module,
            n_steps=raw_budget,
            seed=seed,
            device=device,
            warmup=0,
        )
        scored = np.flatnonzero(raw.episode_steps >= WARMUP)
        if len(scored) >= n_steps:
            break
        raw_budget *= 2
    if raw.episode_ids is None:
        raise AssertionError("geometry collection requires explicit episode IDs")
    stop = int(scored[n_steps - 1]) + 1
    mask = np.zeros(stop, dtype=bool)
    mask[scored[:n_steps]] = True
    return GeometryData(
        activations=raw.activations[:stop].copy(),
        beliefs=raw.joint_beliefs[:stop].copy(),
        observations=raw.observations[:stop].copy(),
        episode_ids=raw.episode_ids[:stop].copy(),
        episode_steps=raw.episode_steps[:stop].copy(),
        mask=mask,
    )


def _history_order(data: GeometryData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.asarray(data.observations)
    episode_ids = np.asarray(data.episode_ids)
    steps = np.asarray(data.episode_steps)
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError("observations must have shape (n, 2)")
    count = len(observations)
    if count == 0:
        raise ValueError("replay requires nonempty complete episode histories")
    for name, values in (("episode_ids", episode_ids), ("episode_steps", steps)):
        if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be an aligned integer vector")
    _, groups = np.unique(episode_ids, return_inverse=True)
    order = np.argsort(groups, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1]
    lengths = np.diff(np.r_[starts, count])
    expected = np.arange(count) - np.repeat(starts, lengths)
    if not np.array_equal(steps[order], expected):
        raise ValueError(
            "each episode history must start at step 0 and advance chronologically by 1"
        )
    resets = steps == 0
    if not np.all(observations[resets] == 0):
        raise ValueError("reset observations must be blank")
    visible = observations[~resets]
    if not np.all((visible == 0) | (visible == 1)) or not np.all(
        visible.sum(axis=1) == 1
    ):
        raise ValueError("nonreset observations must be visible-token one-hots")
    return order, groups, observations.argmax(axis=1)


def _condition_sources(
    sources: np.ndarray,
    edges: np.ndarray,
    tokens: np.ndarray,
) -> np.ndarray:
    posterior = (sources[:, None, :] @ edges[tokens])[:, 0, :]
    total = posterior.sum(axis=1, keepdims=True)
    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        raise ValueError("impossible observation produced zero probability mass")
    return posterior / total


def replay_beliefs(
    data: GeometryData,
    *,
    alpha: float = 0.97,
    t0: float = 0.38,
    t1: float = 0.54,
    suffix_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if suffix_length is not None and (
        isinstance(suffix_length, (bool, np.bool_))
        or not isinstance(suffix_length, (int, np.integer))
        or suffix_length < 0
    ):
        raise ValueError("suffix_length must be a nonnegative integer or None")
    order, groups, tokens = _history_order(data)
    steps = np.asarray(data.episode_steps)
    model = strata_model(alpha=alpha, t0=t0, t1=t1)
    edges = model.edge_transition_matrices
    sources = np.broadcast_to(model.initial_distribution, (len(steps), 3)).copy()
    longest = int(steps.max())
    if suffix_length is None or suffix_length >= longest:
        current = np.broadcast_to(
            model.initial_distribution, (int(groups.max()) + 1, 3)
        ).copy()
        step_order = np.argsort(steps, kind="stable")
        boundaries = np.flatnonzero(np.diff(steps[step_order])) + 1
        for rows in np.split(step_order, boundaries)[1:]:
            members = groups[rows]
            current[members] = _condition_sources(current[members], edges, tokens[rows])
            sources[rows] = current[members]
    else:
        positions = np.empty(len(steps), dtype=np.int64)
        positions[order] = np.arange(len(steps))
        for lag in range(int(suffix_length) - 1, -1, -1):
            rows = np.flatnonzero(steps > lag)
            visible_rows = order[positions[rows] - lag]
            sources[rows] = _condition_sources(sources[rows], edges, tokens[visible_rows])
    return sources @ model.transition_matrix, sources @ model.emission_matrix


__all__ = ["GeometryData", "collect_geometry_data", "replay_beliefs"]
