from harness.context import RunContext

from experiments.strata_token_guess_cycle_2 import shared


def build_config(context: RunContext):
    return shared.build_config(context)


def run(context: RunContext):
    return shared.run_condition(context)
