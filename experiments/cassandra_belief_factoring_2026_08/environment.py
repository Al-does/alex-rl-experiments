"""Experiment-local policy observation adapter for Cassandra maintenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from envs.cassandra_machine import CassandraMachineEnv
from envs.cassandra_machine.model import Action, N_OBSERVATIONS


OBSERVATION_DIM = N_OBSERVATIONS + len(Action)


class CassandraActionObservationEnv(gym.Wrapper):
    """Expose the current symbol and preceding action as one-hot features.

    Cassandra's Bayesian filter is action-conditioned. A transformer that only
    receives observation symbols cannot reconstruct that filter when PPO
    samples actions stochastically. This adapter keeps the canonical symbol
    observation while making the agent's own preceding action explicit.

    Rewards are deliberately not included: the environment's public
    ``belief_current`` diagnostic conditions on actions and observations, not
    on the state-informative operate reward.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        values["observation_mode"] = "symbol"
        environment = CassandraMachineEnv(values)
        super().__init__(environment)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

    @staticmethod
    def encode(symbol: int, previous_action: int | None) -> np.ndarray:
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[int(symbol)] = 1.0
        if previous_action is not None:
            observation[N_OBSERVATIONS + int(previous_action)] = 1.0
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        symbol, info = self.env.reset(seed=seed, options=options)
        return self.encode(int(symbol), None), info

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        symbol, reward, terminated, truncated, info = self.env.step(action)
        return (
            self.encode(int(symbol), int(info["action"])),
            reward,
            terminated,
            truncated,
            info,
        )
