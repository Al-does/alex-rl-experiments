"""Persistent-nuisance ladder with the PR #195 training recipe."""
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import shared as source
from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.training import run_study

STUDY = "belief_quotient_persistent_nuisance_jax_2026_11"
SOURCE_STUDY = source.STUDY
ENVIRONMENT_REVISION = "persistent_nuisance_v1"
ARCHITECTURES = source.ARCHITECTURES
CONDITION_NAMES = source.CONDITION_NAMES
TOTAL_ENV_STEPS = source.TOTAL_ENV_STEPS
BASELINE_TRANSITION_MATRIX = source.BASELINE_TRANSITION_MATRIX.copy()
EMISSION_MATRIX = np.array([[.1875, .0625, .75], [.0625, .1875, .75], [0., 0., 1.]])
EFFECT_SIZE = source.EFFECT_SIZE
DIRECTIONS = source.DIRECTIONS
REWARD_STATE = source.REWARD_STATE
train_config = source.train_config
model_spec = source.model_spec
belief_targets = source.belief_targets
tilt_reward_state = source.tilt_reward_state


def emission_matrix():
    return EMISSION_MATRIX.copy()


def action_transitions(variant):
    if variant not in DIRECTIONS:
        raise ValueError("variant must be 1, 2, or 3")
    return np.stack([BASELINE_TRANSITION_MATRIX] + [
        tilt_reward_state(BASELINE_TRANSITION_MATRIX, DIRECTIONS[variant][:, action])
        for action in (1, 2)
    ])


def ladder_process(variant):
    reward = (np.arange(3) == REWARD_STATE).astype(float)
    return make_process(np.full(3, 1 / 3), action_transitions(variant), emission_matrix(),
                        reward, reward[None])


def environment_metadata():
    return {
        "source_study": SOURCE_STUDY,
        "environment_revision": ENVIRONMENT_REVISION,
        "baseline_transition_matrix": BASELINE_TRANSITION_MATRIX.tolist(),
        "emission_matrix": EMISSION_MATRIX.tolist(),
        "effect_size": EFFECT_SIZE,
        "directions": {str(k): v.tolist() for k, v in DIRECTIONS.items()},
    }


def run_condition(context, *, variant, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    process = ladder_process(variant)
    return run_study(
        context, study=STUDY, condition=f"variant_{variant}", architecture=architecture,
        process=process, spec=model_spec(architecture, process),
        config=train_config(context.smoke), targets=belief_targets, representation="embedding",
        recipe_extra={**environment_metadata(), "variant": variant,
                      "condition_name": CONDITION_NAMES[variant],
                      "reward": "pre-transition state-2 occupancy",
                      "representation": "post_final_layer_norm"},
    )
