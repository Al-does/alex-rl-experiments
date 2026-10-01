"""E4 deliberate reactive-sufficient null at alpha1=0.85."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e4_gauge_a085_reactive_null", experiment="E4", action_kind="e4_gauge",
    reward_kind="f2_goal", alpha1=0.85, alpha2=0.85,
    expected_quotient_dimension=None, campaign_role="reactive_sufficient_null",
    hypothesis="F1 symbols matter, but no F1 belief geometry should be demanded.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
