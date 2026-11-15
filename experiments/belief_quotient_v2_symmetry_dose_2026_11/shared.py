"""Graded symmetry breaking of the Reward-State Quotient in the persistent-nuisance v2 environment.

Every arm is the ``belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11`` transformer recipe
(2B steps, T=1.5, constant entropy 0, epsilon 0) with one environment knob changed:

* ``lam`` interpolates state 2's tilt directions (manuscript state names): Tilt X =
  (+1, lam, -1), Tilt Y = (-1, -lam, -1). ``lam=1`` is exactly the Reward-State Quotient and
  ``lam=-1`` exactly the Full-State Quotient.
* ``kappa`` breaks the state-1/state-2 symmetry in the emissions only: P(C | state 1) =
  0.85 - kappa, P(C | state 2) = 0.85 + kappa, keeping each state's 3:1 A/B split.
"""
from dataclasses import replace

import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.training import run_study
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2

STUDY = "belief_quotient_v2_symmetry_dose_2026_11"
ENVIRONMENT_REVISION = "persistent_nuisance_v2_symmetry_dose"


def directions(lam):
    return np.array([[0., 1., -1.], [0., lam, -lam], [0., -1., -1.]])


def action_transitions(lam):
    if not -1 <= lam <= 1:
        raise ValueError("lam must lie in [-1, 1]")
    tilts = directions(lam)
    return np.stack([v2.BASELINE_TRANSITION_MATRIX] + [
        base.tilt_reward_state(v2.BASELINE_TRANSITION_MATRIX, tilts[:, action], effect_size=v2.EFFECT_SIZE)
        for action in (1, 2)])


def emission_matrix(kappa):
    if not 0 <= kappa < .15:
        raise ValueError("kappa must lie in [0, 0.15)")
    return np.array([[(.15 + kappa) * .75, (.15 + kappa) * .25, .85 - kappa],
                     [(.15 - kappa) * .25, (.15 - kappa) * .75, .85 + kappa], [0., 0., 1.]])


def dose_process(lam, kappa):
    reward = (np.arange(3) == v2.REWARD_STATE).astype(float)
    return make_process(v2.INITIAL_STATE_DISTRIBUTION, action_transitions(lam), emission_matrix(kappa),
                        reward, reward[None])


def run_condition(context, *, condition, lam, kappa):
    process = dose_process(lam, kappa)
    config = replace(base.train_config(context.smoke, TOTAL_ENV_STEPS),
                     entropy_schedule=tuple(ENTROPY_SCHEDULE), temperature=TEMPERATURE,
                     exploration_epsilon=0.0)
    return run_study(
        context, study=STUDY, condition=condition, architecture="transformer", process=process,
        spec=base.model_spec("transformer", process), config=config, targets=base.belief_targets,
        representation="embedding",
        recipe_extra={
            "source_study": "belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11",
            "environment_revision": ENVIRONMENT_REVISION, "lam": lam, "kappa": kappa,
            "directions": directions(lam).tolist(),
            "baseline_transition_matrix": v2.BASELINE_TRANSITION_MATRIX.tolist(),
            "emission_matrix": emission_matrix(kappa).tolist(), "effect_size": v2.EFFECT_SIZE,
            "initial_state_distribution": v2.INITIAL_STATE_DISTRIBUTION.tolist(),
            "reward_state": v2.REWARD_STATE, "reward": "pre-transition state-2 occupancy",
            "representation": "post_final_layer_norm",
        },
    )
