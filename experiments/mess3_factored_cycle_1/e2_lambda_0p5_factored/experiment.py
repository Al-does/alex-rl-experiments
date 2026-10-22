"""E2 low-coupling dose point."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e2_lambda_0p5_factored", experiment="E2", action_kind="e2_tilt",
    reward_kind="f2_goal", alpha1=0.85, alpha2=0.65, coupling_lambda=0.5,
    expected_quotient_dimension=1, campaign_role="dose_response",
    hypothesis="F1 signal should track tiny control incentive, not predictive visibility.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
