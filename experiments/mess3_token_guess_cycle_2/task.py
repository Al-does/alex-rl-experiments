"""Rewarded next-token prediction for passive MESS3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent


class NextTokenGuessTask:
    """Reward a guess of the token emitted before the completed transition."""

    requires_belief = False

    def __init__(self, *, model: HMMModel) -> None:
        self.action_space = gym.spaces.Discrete(model.n_tokens)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_tokens,),
            dtype=np.float32,
        )
        self._transition_matrix = model.transition_matrix

    def reset(self) -> None:
        pass

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        guess = int(action)
        if not self.action_space.contains(guess):
            raise ValueError(f"token guess {guess} is outside the action space")
        return ActionDecision(
            requested_action=guess,
            executed_action=guess,
            transition_matrix=self._transition_matrix,
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
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action) - self.action_space.start] = 1.0
        return encoded
