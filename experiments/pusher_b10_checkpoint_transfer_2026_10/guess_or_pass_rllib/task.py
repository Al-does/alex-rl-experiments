from __future__ import annotations

from numbers import Integral, Real

import gymnasium as gym
import numpy as np
from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from envs.hmm.belief import condition_edge, measure, predict


class GuessOrPassTask:
    requires_belief = False

    def __init__(self, *, model: HMMModel, cost: float) -> None:
        if (
            isinstance(cost, bool)
            or not isinstance(cost, Real)
            or not np.isfinite(cost)
            or cost <= 0.0
        ):
            raise ValueError("cost must be a finite positive number")
        self.action_space = gym.spaces.Discrete(3)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.action_space.n,),
            dtype=np.float32,
        )
        self._model = model
        self._cost = float(cost)
        self._transition_matrix = model.transition_matrix
        self._source_belief = model.initial_distribution.copy()

    def reset(self) -> None:
        self._source_belief = self._model.initial_distribution.copy()

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("guess-or-pass action must be an integer")
        selected = int(action)
        if not self.action_space.contains(selected):
            raise ValueError(f"guess-or-pass action {selected} is invalid")
        return ActionDecision(
            requested_action=selected,
            executed_action=selected,
            transition_matrix=self._transition_matrix,
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        token = event.raw_token_before
        if token is None:
            raise ValueError("guess-or-pass requires a pending token")

        token_probabilities = self._source_belief @ self._model.emission_matrix
        max_probability = float(token_probabilities.max())
        bayes_expected_reward = max(
            0.0,
            max_probability - self._cost * (1.0 - max_probability),
        )

        action = int(decision.executed_action)
        passed = float(action == 2)
        guess_correct = float(action != 2 and action == token)
        guess_wrong = float(action != 2 and action != token)
        reward = (
            0.0
            if passed
            else 1.0
            if guess_correct
            else -self._cost
        )

        if self._model.edge_transition_matrices is not None:
            edge_matrices = (
                decision.edge_transition_matrices
                if decision.edge_transition_matrices is not None
                else self._model.edge_transition_matrices
            )
            self._source_belief = condition_edge(
                self._source_belief,
                edge_matrices,
                int(token),
            )
        else:
            posterior = measure(
                self._source_belief,
                self._model.emission_matrix,
                int(token),
            )
            self._source_belief = predict(
                posterior,
                decision.transition_matrix,
            )

        return reward, {
            "guess_correct": guess_correct,
            "guess_wrong": guess_wrong,
            "passed": passed,
            "bayes_expected_reward": bayes_expected_reward,
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(self.action_space.n, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
