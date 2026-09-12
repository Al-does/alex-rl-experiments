from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.alpha095.process import (
    COMPONENT_PARAMETERS,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.shared import (
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext


def build_config(context: RunContext):
    return _build_config(
        context,
        1,
        component_parameters=COMPONENT_PARAMETERS,
    )


def run(context: RunContext):
    return run_condition(
        context,
        1,
        component_parameters=COMPONENT_PARAMETERS,
    )
