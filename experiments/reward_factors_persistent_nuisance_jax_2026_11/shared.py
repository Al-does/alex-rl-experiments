"""Independent persistent-nuisance factors with PR #195 training."""
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.training import run_study
from experiments.belief_quotient_persistent_nuisance_jax_2026_11.shared import (
    action_transitions, emission_matrix, environment_metadata,
)
from experiments.reward_factors_history_jax_2026_11 import shared as source
from experiments.reward_factors_history_jax_2026_11.process import (
    ACTION_PAIRS, CONDITIONS, factor_targets,
)

STUDY = "reward_factors_persistent_nuisance_jax_2026_11"
SOURCE_STUDY = source.STUDY
ARCHITECTURES = source.ARCHITECTURES
TOTAL_ENV_STEPS = source.TOTAL_ENV_STEPS
train_config = source.train_config
model_spec = source.model_spec


def factor_process(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    factor = action_transitions(3)
    transitions = np.stack([np.kron(factor[a], factor[b]) for a, b in ACTION_PAIRS])
    emission = np.kron(emission_matrix(), emission_matrix())
    first, second = np.divmod(np.arange(9), 3)
    components = np.stack([first == 2, second == 2]).astype(float)
    reward = components[0] + (components[1] if condition == "reward_both" else 0.)
    return make_process(np.full(9, 1 / 9), transitions, emission, reward, components)


def run_condition(context, *, condition, architecture):
    if condition not in CONDITIONS or architecture not in ARCHITECTURES:
        raise ValueError("unknown condition or architecture")
    process = factor_process(condition)
    return run_study(
        context, study=STUDY, condition=condition, architecture=architecture,
        process=process, spec=model_spec(architecture, process),
        config=train_config(context.smoke), targets=factor_targets, representation="residual",
        recipe_extra={**environment_metadata(), "source_study": SOURCE_STUDY,
                      "action_pairs": ACTION_PAIRS,
                      "condition_name": source.CONDITION_NAMES[condition],
                      "reward": "factor-1 state-2 occupancy, plus factor 2 for reward_both",
                      "representation": "pre_final_layer_norm"},
    )
