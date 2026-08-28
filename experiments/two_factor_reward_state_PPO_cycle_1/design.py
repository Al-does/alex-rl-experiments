"""Reference policies and the shared preregistered partial-observability audit."""

# The task dynamics and rewards are intentionally identical to PR 65. Reusing
# its exact audit prevents the PPO comparison from drifting scientifically.
from experiments.two_factor_reward_state_SAC_cycle_1.design import (
    AUDIT_BURN_IN,
    AUDIT_CHAINS,
    AUDIT_STEPS,
    GAMMA,
    MAXIMUM_STANDARD_ERROR,
    MINIMUM_DEMAND_GAP,
    REFERENCE_VALUES,
    constant_occupancies,
    controlled_transitions,
    demand_audit,
    fully_observed_occupancy,
    fully_observed_q_values,
)


__all__ = [
    "AUDIT_BURN_IN",
    "AUDIT_CHAINS",
    "AUDIT_STEPS",
    "GAMMA",
    "MAXIMUM_STANDARD_ERROR",
    "MINIMUM_DEMAND_GAP",
    "REFERENCE_VALUES",
    "constant_occupancies",
    "controlled_transitions",
    "demand_audit",
    "fully_observed_occupancy",
    "fully_observed_q_values",
]
