"""E1 primary: independent factors, shared shifts, F2-only reward."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e1_f2_diagonal_factored", experiment="E1", action_kind="diagonal",
    reward_kind="f2_goal", alpha1=0.55, alpha2=0.55,
    expected_quotient_dimension=2,
    hypothesis="Only the F2 belief simplex is reward-relevant.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
