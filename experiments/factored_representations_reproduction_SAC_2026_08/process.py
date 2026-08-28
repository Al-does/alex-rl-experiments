"""Independent MESS3 factors with fixed-window observations for SAC."""

from __future__ import annotations

from typing import Any

from experiments.factored_representations_reproduction_PPO_2026_08.process import (
    FACTOR_CARDINALITY,
    FACTOR_COUNTS,
    FACTOR_STATE_DIMENSION,
    MESS3_ALPHA,
    MESS3_X,
    MESS3_Y,
    PAPER_TRANSITION_MATRIX,
    decode_joint_tokens,
    encode_joint_tokens,
    joint_token_count,
    paper_mess3_model,
)

CONTEXT_LENGTH = 9


def factor_specifications(factor_count: int) -> list[dict[str, Any]]:
    """Return serializable factor factories rooted in this SAC package."""

    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    return [
        {
            "factory": (
                "experiments.factored_representations_reproduction_SAC_2026_08."
                "process:paper_mess3_model"
            )
        }
        for _ in range(factor_count)
    ]


def environment_config(factor_count: int) -> dict[str, Any]:
    """Expose the complete token history to SAC's stateless split networks.

    RLlib's SAC RLModule does not support recurrent state. The generic HMM
    environment therefore emits its existing newest-first token-history window.
    This carries exactly the BOS-plus-eight-token context used by the PPO
    transformer without leaking actions or privileged process state.
    """

    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {"factors": factor_specifications(factor_count)},
        },
        "task": {
            "class": (
                "experiments.factored_representations_reproduction_SAC_2026_08."
                "task:NextJointTokenGuessTask"
            )
        },
        "observation": {
            "token": {"depth": CONTEXT_LENGTH},
            "action": None,
        },
        "delay": 1,
        "episode_length": CONTEXT_LENGTH,
        "randomize_first_episode_length": True,
    }
