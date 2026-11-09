from experiments.reward_factors_history_REINFORCE_2026_11_x5.shared import (
    build_config as _build_config,
    run_condition,
)

CONDITION = "reward_both"
ARCHITECTURE = "transformer"


def build_config(context):
    return _build_config(context, CONDITION, architecture=ARCHITECTURE)


def run(context):
    return run_condition(context, CONDITION, architecture=ARCHITECTURE)
