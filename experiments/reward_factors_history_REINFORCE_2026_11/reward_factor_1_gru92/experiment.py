from experiments.reward_factors_history_REINFORCE_2026_11.shared import (
    build_config as _build_config,
    run_condition,
)

CONDITION = "reward_factor_1"
ARCHITECTURE = "gru92"


def build_config(context):
    return _build_config(context, CONDITION, architecture=ARCHITECTURE)


def run(context):
    return run_condition(context, CONDITION, architecture=ARCHITECTURE)
