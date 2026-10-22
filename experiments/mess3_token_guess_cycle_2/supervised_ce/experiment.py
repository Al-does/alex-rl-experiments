"""Next-token cross-entropy control with no reinforcement-learning loss."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "supervised_ce")
