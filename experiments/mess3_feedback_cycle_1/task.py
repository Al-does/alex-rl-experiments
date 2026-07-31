"""Guessing the composite token, where the guess also steers the register."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.mess3_feedback_cycle_1.dynamics import (
    chain_factor,
    composite_token,
    joint_transitions,
)


class FeedbackTokenGuessTask:
    """Reward a guess of the composite token that the guess itself shifts.

    The reward is the passive token-guess reward, so the myopic optimum is
    unchanged: name the most likely current composite token. Only the executed
    kernel depends on the guess, which makes the feedback a modelling problem
    rather than a control problem under ``gamma = 0``.
    """

    requires_belief = False

    def __init__(
        self,
        *,
        model: HMMModel,
        feedback_strength: float = 0.0,
    ) -> None:
        if not 0.0 <= float(feedback_strength) <= 1.0:
            raise ValueError("feedback_strength must lie in [0, 1]")
        self.feedback_strength = float(feedback_strength)
        self.n_guesses = int(round(np.sqrt(model.n_states)))
        if self.n_guesses**2 != model.n_states:
            raise ValueError(
                "the composed generator must have a square state product"
            )
        if model.n_tokens != model.n_states:
            raise ValueError(
                "the paired token alphabet must match the state product"
            )
        chain = chain_factor(model.transition_matrix, size=self.n_guesses)
        self._transitions = joint_transitions(
            self.feedback_strength,
            base=chain,
            n_actions=self.n_guesses,
        )
        self.action_space = gym.spaces.Discrete(self.n_guesses)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_guesses,),
            dtype=np.float32,
        )

    @property
    def transitions(self) -> np.ndarray:
        """Return every guess-conditioned joint kernel, stacked by guess."""

        return self._transitions

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
            transition_matrix=self._transitions[guess],
            metadata={"feedback_strength": self.feedback_strength},
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        scored = composite_token(event.raw_token_before, size=self.n_guesses)
        correct = float(decision.executed_action == scored)
        return correct, {
            "token_guess_correct": correct,
            "token_guess_scored": 1.0,
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action) - self.action_space.start] = 1.0
        return encoded
