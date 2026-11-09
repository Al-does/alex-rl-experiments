import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision
from experiments.belief_quotient_history_REINFORCE_2026_11.design import (
    BASELINE_TRANSITION_MATRIX,
    EFFECT_SIZE,
    controlled_model,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.task import (
    ActionSymmetryTask,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import (
    ACTION_PAIRS,
    CONDITIONS,
)


class HistoryRewardFactorsTask:
    requires_belief = False

    def __init__(self, *, model, condition):
        if condition not in CONDITIONS or model.n_states != 9:
            raise ValueError("two-factor control requires a valid condition and nine states")
        baseline = np.asarray(BASELINE_TRANSITION_MATRIX)
        if not np.allclose(model.transition_matrix, np.kron(baseline, baseline)):
            raise ValueError("factor model and controlled transitions must use the same baseline")
        self.condition = condition
        self.action_space = gym.spaces.Discrete(len(ACTION_PAIRS))
        self.action_observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(ACTION_PAIRS),), dtype=np.float32
        )
        factor = ActionSymmetryTask(
            model=controlled_model(), variant=3, effect_size=EFFECT_SIZE
        )
        self._transitions = tuple(
            np.kron(
                factor.transition_matrix_for_action(first),
                factor.transition_matrix_for_action(second),
            )
            for first, second in ACTION_PAIRS
        )
        for transition in self._transitions:
            transition.setflags(write=False)

    def reset(self):
        pass

    def transition_matrix_for_action(self, action):
        index = int(action)
        if not self.action_space.contains(index):
            raise ValueError("action must lie in [0, 9)")
        return self._transitions[index]

    def resolve_action(self, action, state, model):
        executed = int(action)
        return ActionDecision(
            requested_action=executed,
            executed_action=executed,
            transition_matrix=self.transition_matrix_for_action(executed),
        )

    def reward(self, event, decision):
        first, second = divmod(event.state_before, 3)
        first_reward = float(first == 2)
        second_reward = float(second == 2)
        return (
            first_reward + (second_reward if self.condition == "reward_both" else 0.0),
            {
                "factor_1_occupancy_reward": first_reward,
                "factor_2_occupancy_reward": second_reward,
            },
        )

    def encode_action(self, executed_action):
        encoded = np.zeros(len(ACTION_PAIRS), dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
