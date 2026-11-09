from experiments.belief_quotient_history_REINFORCE_2026_11.shared import (
    build_config as _build_config,
    run_condition,
)

VARIANT = 3
ARCHITECTURE = "gru92"


def build_config(context):
    return _build_config(context, VARIANT, architecture=ARCHITECTURE)


def run(context):
    return run_condition(context, VARIANT, architecture=ARCHITECTURE)
