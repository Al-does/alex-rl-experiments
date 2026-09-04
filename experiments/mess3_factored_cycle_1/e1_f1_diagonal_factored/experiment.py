"""E1 reward-site swap control."""

from harness.context import RunContext
from experiments.mess3_factored_cycle_1.shared import Condition, build_config as _build, run_condition

CONDITION = Condition(
    name="e1_f1_diagonal_factored", experiment="E1", action_kind="diagonal",
    reward_kind="f1_goal", alpha1=0.55, alpha2=0.55,
    expected_quotient_dimension=2, campaign_role="reward_site_swap",
    hypothesis="Swapping the reward site swaps which factor simplex is retained.",
)

def build_config(context: RunContext):
    return _build(context, CONDITION)

def run(context: RunContext):
    return run_condition(context, CONDITION)
