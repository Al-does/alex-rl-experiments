"""Experiment-local next-token tasks for Kelly reward conditions."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.mess_3_kelly_cycle_1.kelly import (
    MAX_WAGER,
    kelly_fraction,
    realized_log_growth,
)


class RawNextTokenTask:
    """Expose correctness so the Learner can form policy-dependent rewards."""

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


class BayesOracleKellyTask(RawNextTokenTask):
    """Size each selected-token wager from the exact hidden Bayesian filter."""

    requires_belief = True

    def __init__(self, *, model: HMMModel) -> None:
        super().__init__(model=model)
        self._emission_matrix = model.emission_matrix

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        if event.belief_before is None:
            raise RuntimeError("Bayes-oracle Kelly reward requires belief tracking")
        guess = int(decision.executed_action)
        probability = float(
            event.belief_before @ self._emission_matrix[:, guess]
        )
        wager = float(kelly_fraction(probability, max_wager=MAX_WAGER))
        correct = float(guess == event.raw_token_before)
        growth = float(realized_log_growth(correct, wager))
        return growth, {
            "kelly_log_growth": growth,
            "kelly_wager": wager,
            "kelly_probability": probability,
            "token_guess_correct": correct,
            "token_guess_scored": 1.0,
        }
