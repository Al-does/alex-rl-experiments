"""Action and reward semantics for the two-factor MESS3 conditions."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.mess3_factored_cycle_1.dynamics import (
    N_FACTOR_STATES,
    N_JOINT_STATES,
    action_kernels,
    reward_vector,
)


class FactoredControlTask:
    """Execute one action-conditioned kernel and reward current occupancy."""

    requires_belief = False

    def __init__(
        self,
        *,
        model: HMMModel,
        action_kind: str,
        reward_kind: str,
        coupling_lambda: float = 0.0,
        action_encoding: str = "factored",
    ) -> None:
        if model.n_states != N_JOINT_STATES or model.n_tokens != N_JOINT_STATES:
            raise ValueError("factored MESS3 requires nine states and nine symbols")
        if action_encoding not in {"factored", "joint"}:
            raise ValueError("action_encoding must be 'factored' or 'joint'")
        self.action_kind = str(action_kind)
        self.reward_kind = str(reward_kind)
        self.coupling_lambda = float(coupling_lambda)
        self.action_encoding = action_encoding
        self._kernels = action_kernels(
            self.action_kind,
            coupling_lambda=self.coupling_lambda,
        )
        self._rewards = reward_vector(self.reward_kind)
        self.action_space = gym.spaces.Discrete(len(self._kernels))
        width = (
            2 * N_FACTOR_STATES
            if self.action_kind == "product" and action_encoding == "factored"
            else len(self._kernels)
        )
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(width,),
            dtype=np.float32,
        )

    @property
    def kernels(self) -> np.ndarray:
        """Return immutable-by-convention action kernels for diagnostics."""

        return self._kernels

    def reset(self) -> None:
        pass

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        executed = int(action)
        if not self.action_space.contains(executed):
            raise ValueError(f"action {executed} is outside the action space")
        return ActionDecision(
            requested_action=executed,
            executed_action=executed,
            transition_matrix=self._kernels[executed],
            metadata={
                "action_kind": self.action_kind,
                "coupling_lambda": self.coupling_lambda,
            },
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        reward = float(self._rewards[event.state_before])
        first, second = divmod(event.state_before, N_FACTOR_STATES)
        return reward, {
            "reward": reward,
            "f1_goal_occupancy": float(first == 2),
            "f2_goal_occupancy": float(second == 2),
            "joint_goal_occupancy": float(first == 2 and second == 2),
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        action = int(executed_action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} is outside the action space")
        if self.action_kind == "product" and self.action_encoding == "factored":
            first, second = divmod(action, N_FACTOR_STATES)
            encoded = np.zeros(2 * N_FACTOR_STATES, dtype=np.float32)
            encoded[first] = 1.0
            encoded[N_FACTOR_STATES + second] = 1.0
            return encoded
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[action] = 1.0
        return encoded
