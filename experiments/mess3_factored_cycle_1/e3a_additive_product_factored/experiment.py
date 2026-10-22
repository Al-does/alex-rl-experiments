"""E3a: additive reward and independent product actions."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e3a_additive_product_factored", experiment="E3a", action_kind="product",
    reward_kind="additive", alpha1=0.55, alpha2=0.55,
    expected_quotient_dimension=4,
    hypothesis="Both beliefs are needed, while policy and value remain factor-separable.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
