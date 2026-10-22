"""E1 product-action control with an indifferent F1 action component."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e1_f2_product_factored", experiment="E1", action_kind="product",
    reward_kind="f2_goal", alpha1=0.55, alpha2=0.55,
    expected_quotient_dimension=2, campaign_role="product_action_ablation",
    hypothesis="F1 remains irrelevant despite a formally free F1 action component.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
