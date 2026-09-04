"""E1 joint-symbol/joint-action member of the asymmetric encoding pair."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e1_product_joint_asymmetric", experiment="E1", action_kind="product",
    reward_kind="f2_goal", alpha1=0.60, alpha2=0.55,
    token_encoding="joint", action_encoding="joint",
    expected_quotient_dimension=2, campaign_role="encoding_ablation",
    hypothesis="Tests discovery of factors from nine-way symbol and action alphabets.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
