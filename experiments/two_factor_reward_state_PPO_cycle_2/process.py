"""Recurrent PPO view of the shared cycle-2 two-factor process."""

from __future__ import annotations

from typing import Any

from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    FACTOR_CARDINALITY,
    FACTOR_COUNT,
    JOINT_STATE_COUNT,
    JOINT_TOKEN_COUNT,
    LOCAL_CONTEXT_LENGTH,
    MESS3_ALPHA,
    TRANSFORMER_LAYERS,
    TRANSFORMER_LOOKBACK,
    TRANSITION_MATRIX,
    decode_joint_indices,
    factor_specifications,
)


def environment_config(condition: str) -> dict[str, Any]:
    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {"factors": factor_specifications()},
        },
        "task": {
            "class": (
                "experiments.two_factor_reward_state_SAC_cycle_2.task:"
                "TwoFactorVariant3Task"
            ),
            "kwargs": {"condition": condition},
        },
        "observation": {
            "token": {"depth": 1},
            "action": {"depth": 1},
        },
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }
