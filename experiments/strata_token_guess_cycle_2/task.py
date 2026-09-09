from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent


class StrataTokenGuessTask:
    """Reward the action whose index matches the pending Strata token."""

    requires_belief = False

    def __init__(self, *, model: HMMModel) -> None:
        if model.edge_transition_matrices is None:
            raise ValueError("Strata token guessing requires an edge-emitting model")
        self._model = model
        self.action_space = gym.spaces.Discrete(model.n_tokens)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_tokens,),
            dtype=np.float32,
        )

    def reset(self) -> None:
        pass

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state
        if model is not self._model:
            raise ValueError("task must be used with the model it was constructed for")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not self.action_space.contains(action)
        ):
            raise ValueError("token guess is outside the action space")
        guess = int(action)
        return ActionDecision(
            requested_action=guess,
            executed_action=guess,
            transition_matrix=model.transition_matrix,
            edge_transition_matrices=model.edge_transition_matrices,
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        correct = float(decision.executed_action == event.raw_token_before)
        return correct, {
            "token_guess_correct": correct,
            "token_guess_scored": 1.0,
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        if not self.action_space.contains(executed_action):
            raise ValueError("executed token guess is outside the action space")
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
