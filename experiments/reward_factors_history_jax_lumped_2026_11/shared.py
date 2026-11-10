"""JAX reward-factor recipe: two reward conditions x four architectures."""

from __future__ import annotations

import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process

from experiments.belief_quotient_history_jax_2026_11.models import architecture_spec
from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig
from experiments.belief_quotient_history_jax_2026_11.shared import (
    ARCHITECTURES,
    EPISODE_LENGTH,
    NUM_ENVS,
    NUM_MINIBATCHES,
    entropy_schedule,
)
from experiments.belief_quotient_history_jax_2026_11.training import run_study


STUDY = "reward_factors_history_jax_lumped_2026_11"
SOURCE_STUDY = "reward_factors_history_jax_2026_11"
CONDITION_NAMES = {"reward_both": "Joint Reward", "reward_factor_1": "Unrewarded Factor 2"}
TOTAL_ENV_STEPS = 2_000_000_000
SMOKE_ENV_STEPS = 4_096
CONTEXT_LEN = 10


EMISSION_ALPHA = 0.75
BASELINE_TRANSITION_MATRIX = np.array([
    [0.4875, 0.4875, 0.025],
    [0.4875, 0.4875, 0.025],
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
ENVIRONMENT_REVISION = "uniform_nonreward_destination_v2"


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



CONDITIONS = ("reward_both", "reward_factor_1")
NOOP, POSITIVE, NEGATIVE = 0, 1, 2
ACTION_PAIRS = (
    (NOOP, NOOP), (POSITIVE, POSITIVE), (POSITIVE, NEGATIVE),
    (NOOP, POSITIVE), (NOOP, NEGATIVE), (NEGATIVE, POSITIVE),
    (NEGATIVE, NEGATIVE), (POSITIVE, NOOP), (NEGATIVE, NOOP),
)


def factor_process(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    factor = action_transitions(3)
    transitions = np.stack([np.kron(factor[a], factor[b]) for a, b in ACTION_PAIRS])
    emission = np.kron(emission_matrix(), emission_matrix())
    first, second = np.divmod(np.arange(9), 3)
    components = np.stack([(first == 2), (second == 2)]).astype(float)
    reward = components[0] + (components[1] if condition == "reward_both" else 0.0)
    return make_process(np.full(9, 1 / 9), transitions, emission, reward, components)


def factor_targets(belief):
    joint = belief.reshape(-1, 3, 3)
    return {"joint": belief, "factor_1": joint.sum(2), "factor_2": joint.sum(1)}


def train_config(smoke: bool, total_env_steps: int = TOTAL_ENV_STEPS,
                 entropy_schedule_points=None) -> TrainConfig:
    return TrainConfig(total_env_steps=SMOKE_ENV_STEPS if smoke else total_env_steps,
                       episode_length=EPISODE_LENGTH, num_envs=NUM_ENVS, num_minibatches=NUM_MINIBATCHES,
                       entropy_schedule=entropy_schedule() if entropy_schedule_points is None else tuple(entropy_schedule_points))


def model_spec(architecture, process):
    return architecture_spec(architecture, obs_dim=process.obs_dim, n_actions=process.n_actions,
                             n_tokens=process.n_tokens, context_len=CONTEXT_LEN)


def run_condition(context, *, condition, architecture, total_env_steps=TOTAL_ENV_STEPS, study=STUDY,
                  continue_from=None, entropy_schedule_points=None):
    if condition not in CONDITIONS or architecture not in ARCHITECTURES:
        raise ValueError("unknown condition or architecture")
    process = factor_process(condition)
    return run_study(
        context, study=study, condition=condition, architecture=architecture,
        process=process, spec=model_spec(architecture, process),
        config=train_config(context.smoke, total_env_steps, entropy_schedule_points), targets=factor_targets,
        representation="residual", continue_from=continue_from,
        recipe_extra={
            "source_study": SOURCE_STUDY,
            "environment_revision": ENVIRONMENT_REVISION,
            "factor_baseline_transition_matrix": BASELINE_TRANSITION_MATRIX.tolist(),
            "condition_name": CONDITION_NAMES[condition],
            "reward": "factor-1 occupancy of state 2, plus factor 2 for reward_both",
            "representation": "pre_final_layer_norm",
        },
    )
