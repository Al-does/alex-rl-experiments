from harness.context import RunContext

from experiments.pusher_b10_discrete_SAC_2026_09.shared import (
    build_config as _build_config,
    run_condition,
)


def build_config(context: RunContext):
    return _build_config(context, architecture="split_transformers")


def run(context: RunContext):
    return run_condition(context, architecture="split_transformers")
