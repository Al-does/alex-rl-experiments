"""Two-factor Full-State Quotient control with per-factor reward states."""

from __future__ import annotations

from typing import Any, Sequence

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.two_factor_reward_state_PPO_cycle_2.process import (
    environment_config as source_environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    FACTOR_CARDINALITY,
    FACTOR_COUNT,
    decode_joint_indices,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import (
    REWARD_STATE,
    TwoFactorVariant3Task,
)


SOURCE_REWARD_STATES = (REWARD_STATE, REWARD_STATE)
# Factor 2 now pays for code state 0 (manuscript state 1) instead of 2 (state 3).
MOVED_REWARD_STATES = (REWARD_STATE, 0)


class PerFactorRewardStateTask(TwoFactorVariant3Task):
    """Joint Reward dynamics and actions; each factor pays for its own state."""

    def __init__(self, *, model: HMMModel, reward_states: Sequence[int]) -> None:
        super().__init__(model=model, condition="reward_both")
        states = tuple(int(state) for state in reward_states)
        if len(states) != FACTOR_COUNT or not all(
            0 <= state < FACTOR_CARDINALITY for state in states
        ):
            raise ValueError("reward_states needs one valid state per factor")
        self.reward_states = states

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        factors = decode_joint_indices(event.state_before)
        first, second = (
            float(state == target)
            for state, target in zip(factors, self.reward_states)
        )
        return first + second, {
            "factor_1_occupancy_reward": first,
            "factor_2_occupancy_reward": second,
        }


def environment_config(
    reward_states: Sequence[int] = MOVED_REWARD_STATES,
) -> dict[str, Any]:
    config = source_environment_config("reward_both")
    config["task"] = {
        "class": (
            "experiments.two_factor_moved_reward_steering_2026_11.task:"
            "PerFactorRewardStateTask"
        ),
        "kwargs": {"reward_states": [int(state) for state in reward_states]},
    }
    return config
