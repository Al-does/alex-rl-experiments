"""E2 operating point with a joint nine-symbol presentation."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e2_lambda_1p0_joint", experiment="E2", action_kind="e2_tilt",
    reward_kind="f2_goal", alpha1=0.85, alpha2=0.65, coupling_lambda=1.0,
    token_encoding="joint", action_encoding="joint",
    expected_quotient_dimension=1, campaign_role="encoding_ablation",
    hypothesis="Tests whether the approximate block quotient survives joint symbols.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
