"""JAX ladder recipe: three quotient conditions x four architectures."""

from __future__ import annotations

import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.models import architecture_spec
from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig
from experiments.belief_quotient_history_jax_2026_11.training import run_study

STUDY = "belief_quotient_history_jax_2026_11"
SOURCE_STUDY = "belief_quotient_history_REINFORCE_2026_11"
CONDITION_NAMES = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}
ARCHITECTURES = ("transformer", "memoryless", "gru128", "gru92")
EMISSION_ALPHA = 0.75
BASELINE_TRANSITION_MATRIX = np.array([
    [0.9375, 0.0375, 0.025],
    [0.0375, 0.9375, 0.025],
    [0.075, 0.075, 0.85],
])
EFFECT_SIZE = 1.5
REWARD_STATE = 2
# Column a holds action a's log-odds direction for reward state 2 in each row.
DIRECTIONS = {
    1: np.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, -1.0]]),
    2: np.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]]),
    3: np.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0]]),
}
TOTAL_ENV_STEPS = 10_000_000
ENTROPY_COEFF = 0.015
NUM_MINIBATCHES = 32
SMOKE_ENV_STEPS = 4_096
EPISODE_LENGTH = 1024
CONTEXT_LEN = 32


def emission_matrix(alpha=EMISSION_ALPHA):
    off = (1.0 - alpha) / 2.0
    return np.full((3, 3), off) + np.eye(3) * (alpha - off)


def tilt_reward_state(transition, directions, effect_size=EFFECT_SIZE):
    log_weights = np.log(np.asarray(transition, dtype=np.float64))
    log_weights[:, REWARD_STATE] += effect_size * directions
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=1, keepdims=True)


def action_transitions(variant):
    """Noop keeps the baseline; actions 1 and 2 tilt reward-state odds."""

    if variant not in DIRECTIONS:
        raise ValueError("variant must be 1, 2, or 3")
    directions = DIRECTIONS[variant]
    return np.stack([
        BASELINE_TRANSITION_MATRIX,
        tilt_reward_state(BASELINE_TRANSITION_MATRIX, directions[:, 1]),
        tilt_reward_state(BASELINE_TRANSITION_MATRIX, directions[:, 2]),
    ])


def ladder_process(variant):
    reward = (np.arange(3) == REWARD_STATE).astype(float)
    return make_process(np.full(3, 1 / 3), action_transitions(variant), emission_matrix(),
                        reward, reward[None])


def entropy_schedule():
    """Constant entropy bonus (no annealing)."""

    return ((0, ENTROPY_COEFF),)


def train_config(smoke: bool) -> TrainConfig:
    total = SMOKE_ENV_STEPS if smoke else TOTAL_ENV_STEPS
    return TrainConfig(total_env_steps=total, episode_length=EPISODE_LENGTH,
                       num_minibatches=NUM_MINIBATCHES, entropy_schedule=entropy_schedule())


def model_spec(architecture, process):
    return architecture_spec(architecture, obs_dim=process.obs_dim, n_actions=process.n_actions,
                             n_tokens=process.n_tokens, context_len=CONTEXT_LEN)


def belief_targets(belief):
    return {"belief": belief}


def run_condition(context, *, variant, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    process = ladder_process(variant)
    return run_study(
        context, study=STUDY, condition=f"variant_{variant}", architecture=architecture,
        process=process, spec=model_spec(architecture, process),
        config=train_config(context.smoke), targets=belief_targets,
        representation="embedding",
        recipe_extra={
            "source_study": SOURCE_STUDY,
            "condition_name": CONDITION_NAMES[variant],
            "variant": variant,
            "effect_size": EFFECT_SIZE,
            "emission_alpha": EMISSION_ALPHA,
            "reward": "1 when the pre-transition hidden state is 2",
            "representation": "post_final_layer_norm",
        },
    )
