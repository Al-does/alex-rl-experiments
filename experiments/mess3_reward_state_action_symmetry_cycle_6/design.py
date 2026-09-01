"""Cycle 6 keeps the cycle-5 sticky-state scientific design fixed."""

from experiments.mess3_reward_state_action_symmetry_cycle_5.design import (  # noqa: F401
    CYCLE_5_TRANSITION_MATRIX,
    EFFECT_SIZE,
    EXPECTED_ORACLE_POLICIES,
    analytic_design_summary,
)

CYCLE_6_TRANSITION_MATRIX = CYCLE_5_TRANSITION_MATRIX

__all__ = [
    "CYCLE_6_TRANSITION_MATRIX",
    "EFFECT_SIZE",
    "EXPECTED_ORACLE_POLICIES",
    "analytic_design_summary",
]
