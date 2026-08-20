"""Targeted PPG with canonical cadence and conservative value transfer."""

from harness.context import RunContext
from learners import PPGConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppg_5m.shared import (
    build_config as build_shared_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppg_5m.shared import (
    run_intervention,
)


CONDITION = "targeted_ppg_npi32_value_0p003"
POLICY_ITERATIONS_PER_AUX = 32
AUX_VALUE_LOSS_COEFF = 0.003


def build_config(context: RunContext) -> PPGConfig:
    return build_shared_config(
        context,
        policy_iterations_per_aux=POLICY_ITERATIONS_PER_AUX,
        aux_value_loss_coeff=AUX_VALUE_LOSS_COEFF,
    )


def run(context: RunContext):
    return run_intervention(
        context,
        condition=CONDITION,
        policy_iterations_per_aux=POLICY_ITERATIONS_PER_AUX,
        aux_value_loss_coeff=AUX_VALUE_LOSS_COEFF,
    )


__all__ = ["build_config", "run"]
