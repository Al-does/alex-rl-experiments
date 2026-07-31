"""Token guessing whose guess also steers the MESS3 transition kernel."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.mess3_feedback_cycle_1.dynamics import feedback_transitions


class FeedbackTokenGuessTask:
    """Reward a token guess that also shifts the hidden state it predicts.

    The reward is identical to the passive token-guess task, so the myopic
    optimum is unchanged: guess the most likely current token. Only the
    executed transition kernel depends on the guess, which makes the feedback
    a modelling problem rather than a control problem under ``gamma = 0``.
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
        if model.n_tokens != model.n_states:
            raise ValueError(
                "the guess-driven shift requires one guess per hidden state"
            )
        self.feedback_strength = float(feedback_strength)
        self.action_space = gym.spaces.Discrete(model.n_tokens)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_tokens,),
            dtype=np.float32,
        )
        self._transitions = feedback_transitions(
            self.feedback_strength,
            base=model.transition_matrix,
            n_actions=model.n_tokens,
        )

    @property
    def transitions(self) -> np.ndarray:
        """Return every guess-conditioned kernel stacked by guess."""

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
        correct = float(decision.executed_action == event.raw_token_before)
        return correct, {
            "token_guess_correct": correct,
            "token_guess_scored": 1.0,
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action) - self.action_space.start] = 1.0
        return encoded
