"""E3c: additive reward with one shared diagonal action."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e3c_additive_diagonal_factored", experiment="E3c", action_kind="diagonal",
    reward_kind="additive", alpha1=0.55, alpha2=0.55,
    expected_quotient_dimension=4,
    hypothesis="The shared-action policy itself needs joint factor features.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
