"""E4 primary: noisy F1 gauge controls F2 action semantics."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e4_gauge_a050_factored", experiment="E4", action_kind="e4_gauge",
    reward_kind="f2_goal", alpha1=0.50, alpha2=0.85,
    expected_quotient_dimension=None,
    hypothesis="F1 and relative-phase information are required despite F2-only reward.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
