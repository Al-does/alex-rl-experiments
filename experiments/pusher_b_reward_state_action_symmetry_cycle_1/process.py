from __future__ import annotations

from experiments.pusher_b.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    PRESETS,
    STATE_COUNT,
    TOKEN_COUNT,
    pusher_b_edge_matrices,
    pusher_b_model,
)


EFFECT_SIZE = 1.5
REWARD_STATES = ("A", "B")


def environment_config(
    preset: str,
    variant: int,
    reward_state: str,
) -> dict[str, object]:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    if reward_state not in REWARD_STATES:
        raise ValueError("reward_state must be 'A' or 'B'")
    return {
        "model": {
            "factory": "experiments.pusher_b.process:pusher_b_model",
            "kwargs": {"preset": preset},
        },
        "task": {
            "class": (
                "experiments.pusher_b_reward_state_action_symmetry_cycle_1"
                ".task:ActionSymmetryTask"
            ),
            "kwargs": {
                "variant": variant,
                "reward_state": reward_state,
                "effect_size": EFFECT_SIZE,
            },
        },
        "observation": {
            "token": {"depth": 1},
            "action": {"depth": 1},
        },
        "delay": 1,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": False,
    }


__all__ = [
    "CONTEXT_LENGTH",
    "EFFECT_SIZE",
    "EPISODE_LENGTH",
    "PRESETS",
    "REWARD_STATES",
    "STATE_COUNT",
    "TOKEN_COUNT",
    "environment_config",
    "pusher_b_edge_matrices",
    "pusher_b_model",
]
