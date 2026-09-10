from harness.context import RunContext

from .. import shared


def build_config(context: RunContext):
    return shared.build_config(context)


def run(context: RunContext):
    return shared.run_condition(context)
