"""Experiment-local observation presentations for the same joint HMM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from envs.hmm import HMMEnv
from experiments.mess3_factored_cycle_1.dynamics import N_FACTOR_STATES


class FactoredObservationHMMEnv(gym.ObservationWrapper):
    """Expose a joint symbol as two fixed three-way one-hot slots.

    The wrapped simulator still uses PR 35's ordinary nine-symbol factored
    ``HMMModel``.  Only the policy presentation changes: the first nine joint
    token features become two three-feature marginals.  Previous-action
    features are copied unchanged.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(HMMEnv(config))
        source = self.env.observation_space
        if not isinstance(source, gym.spaces.Box) or len(source.shape) != 1:
            raise TypeError("factored observation wrapper requires a flat Box")
        if self.env.model.n_tokens != N_FACTOR_STATES**2:
            raise ValueError("factored observation wrapper requires nine symbols")
        suffix_width = source.shape[0] - self.env.model.n_tokens
        if suffix_width < 0:
            raise ValueError("source observation is smaller than its token block")
        width = 2 * N_FACTOR_STATES + suffix_width
        self.observation_space = gym.spaces.Box(
            low=np.zeros(width, dtype=np.float32),
            high=np.ones(width, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def model(self):
        return self.env.model

    @property
    def task(self):
        return self.env.task

    @property
    def config(self):
        return self.env.config

    def observation(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        joint = values[: N_FACTOR_STATES**2].reshape(
            N_FACTOR_STATES,
            N_FACTOR_STATES,
        )
        token_features = np.concatenate(
            [joint.sum(axis=1), joint.sum(axis=0)]
        ).astype(np.float32, copy=False)
        return np.concatenate(
            [token_features, values[N_FACTOR_STATES**2 :]]
        ).astype(np.float32, copy=False)
