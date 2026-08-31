"""Experiment-local policy observation adapter for Cassandra maintenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from envs.cassandra_machine import CassandraMachineEnv
from envs.cassandra_machine.model import Action, N_OBSERVATIONS, TargetedAction


OBSERVATION_DIM = N_OBSERVATIONS + len(Action)
TARGETED_OBSERVATION_DIM = N_OBSERVATIONS + len(TargetedAction)


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
        self._n_actions = environment.action_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_OBSERVATIONS + self._n_actions,),
            dtype=np.float32,
        )

    @staticmethod
    def encode(
        symbol: int,
        previous_action: int | None,
        *,
        n_actions: int = len(Action),
    ) -> np.ndarray:
        observation = np.zeros(
            N_OBSERVATIONS + n_actions,
            dtype=np.float32,
        )
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
        return self.encode(
            int(symbol),
            None,
            n_actions=self._n_actions,
        ), info

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        symbol, reward, terminated, truncated, info = self.env.step(action)
        return (
            self.encode(
                int(symbol),
                int(info["action"]),
                n_actions=self._n_actions,
            ),
            reward,
            terminated,
            truncated,
            info,
        )


class CassandraFullyObservablePreviousRewardEnv(gym.Wrapper):
    """Expose exact joint-state one-hot plus preceding scalar reward."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        values["observation_mode"] = "state"
        environment = CassandraMachineEnv(values)
        super().__init__(environment)
        base_space = environment.observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("Cassandra state observations must use a Box space")
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [base_space.low, np.array([-np.inf], dtype=np.float32)]
            ),
            high=np.concatenate(
                [base_space.high, np.array([np.inf], dtype=np.float32)]
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _with_reward(observation: np.ndarray, reward: float) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(observation, dtype=np.float32),
                np.asarray([reward], dtype=np.float32),
            ]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return self._with_reward(observation, 0.0), info

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._with_reward(observation, reward),
            reward,
            terminated,
            truncated,
            info,
        )
