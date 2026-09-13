from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.alpha095.process import (
    COMPONENT_PARAMETERS,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.shared import (
    DEFAULT_ENTROPY_COEFF,
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext

CONTINUATION_ENV_STEPS = 5_000_000
ENTROPY_ANNEAL_SCHEDULE = [[0, DEFAULT_ENTROPY_COEFF], [1_500_000, 0.0]]


def build_config(context: RunContext):
    return _build_config(
        context,
        3,
        component_parameters=COMPONENT_PARAMETERS,
        entropy_coeff=ENTROPY_ANNEAL_SCHEDULE,
    )


def run(context: RunContext):
    return run_condition(
        context,
        3,
        component_parameters=COMPONENT_PARAMETERS,
        total_env_steps=CONTINUATION_ENV_STEPS,
        entropy_coeff=ENTROPY_ANNEAL_SCHEDULE,
        continuation=True,
        recipe_extra={
            "continued_from_run": "nem3-rsa-c5-alpha095-variant_3-s42",
            "continuation_env_steps": CONTINUATION_ENV_STEPS,
            "warm_start_scope": (
                "rl_module weights only; optimizer and step counters reset"
            ),
        },
    )
