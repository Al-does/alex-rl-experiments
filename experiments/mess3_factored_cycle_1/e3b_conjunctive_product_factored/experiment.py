"""E3b: conjunctive reward with product actions."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e3b_conjunctive_product_factored", experiment="E3b", action_kind="product",
    reward_kind="conjunctive", alpha1=0.55, alpha2=0.55,
    expected_quotient_dimension=4,
    hypothesis="The policy factorizes but the value readout needs joint features.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
