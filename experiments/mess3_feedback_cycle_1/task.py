"""Action-conditioned next-token prediction for MESS3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent


class FeedbackNextTokenTask:
    """Guess the pending token, then attract the next state toward that guess."""

    requires_belief = False

    def __init__(self, *, model: HMMModel, eta: float = 0.10) -> None:
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must lie in [0, 1]")
        self.eta = float(eta)
        self.action_space = gym.spaces.Discrete(model.n_tokens)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_tokens,),
            dtype=np.float32,
        )
        baseline = np.asarray(model.transition_matrix, dtype=np.float64)
        transitions = []
        for action in range(model.n_tokens):
            attractor = np.zeros_like(baseline)
            attractor[:, action] = 1.0
            transition = (1.0 - self.eta) * baseline + self.eta * attractor
            transition.setflags(write=False)
            transitions.append(transition)
        self._transitions = tuple(transitions)

    def reset(self) -> None:
        pass

    def transition_matrix_for_action(self, action: int) -> np.ndarray:
        """Return the immutable transition matrix selected by ``action``."""

        guess = int(action)
        if not self.action_space.contains(guess):
            raise ValueError(f"token guess {guess} is outside the action space")
        return self._transitions[guess]

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        guess = int(action)
        return ActionDecision(
            requested_action=guess,
            executed_action=guess,
            transition_matrix=self.transition_matrix_for_action(guess),
            metadata={"feedback_eta": self.eta},
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
