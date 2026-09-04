"""E3a joint-symbol/joint-action member of the asymmetric encoding pair."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e3a_product_joint_asymmetric", experiment="E3a", action_kind="product",
    reward_kind="additive", alpha1=0.60, alpha2=0.55,
    token_encoding="joint", action_encoding="joint",
    expected_quotient_dimension=4, campaign_role="encoding_ablation",
    hypothesis="Tests discovery of both factors without explicit input slots.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
