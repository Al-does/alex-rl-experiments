"""E4 primary dynamics with a joint nine-symbol presentation."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e4_gauge_a050_joint", experiment="E4", action_kind="e4_gauge",
    reward_kind="f2_goal", alpha1=0.50, alpha2=0.85,
    token_encoding="joint", action_encoding="joint",
    expected_quotient_dimension=None, campaign_role="encoding_ablation",
    hypothesis="Gauge and relative-phase information should emerge without fixed slots.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
