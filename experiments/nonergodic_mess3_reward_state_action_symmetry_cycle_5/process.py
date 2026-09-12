from __future__ import annotations

from collections.abc import Mapping, Sequence

from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_COUNT,
    COMPONENT_PARAMETERS,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    TOKEN_COUNT,
    mess3_edge_matrices,
    nonergodic_mess3_model,
)

EFFECT_SIZE = 3.0


def environment_config(
    variant: int,
    *,
    component_parameters: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    model_spec: dict[str, object] = {
        "factory": (
            "experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5"
            ".process:nonergodic_mess3_model"
        ),
    }
    if component_parameters is not None:
        model_spec["kwargs"] = {
            "component_parameters": [
                dict(parameters) for parameters in component_parameters
            ]
        }
    return {
        "model": model_spec,
        "task": {
            "class": (
                "experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5"
                ".task:ActionSymmetryTask"
            ),
            "kwargs": {
                "variant": variant,
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
    "COMPONENT_COUNT",
    "COMPONENT_PARAMETERS",
    "CONTEXT_LENGTH",
    "EFFECT_SIZE",
    "EPISODE_LENGTH",
    "STATE_COUNT",
    "STATES_PER_COMPONENT",
    "TOKEN_COUNT",
    "environment_config",
    "mess3_edge_matrices",
    "nonergodic_mess3_model",
]
