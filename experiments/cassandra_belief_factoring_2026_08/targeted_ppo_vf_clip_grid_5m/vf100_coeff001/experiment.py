"""Train targeted PPO with vf_clip=100 and vf_loss_coeff=0.01."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_vf_clip_grid_5m.shared import (
    VfClipGridCondition,
    build_config as build_grid_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_vf_clip_grid_5m.shared import (
    run_grid_condition,
)
from harness.context import RunContext

CONDITION = VfClipGridCondition(
    name="vf100_coeff001",
    vf_clip_param=100,
    vf_loss_coeff=0.01,
    hypothesis=(
        "Widen value clipping to 10 with a fivefold gentler critic than the failed vf_clip=100 run."
    ),
)


def build_config(context: RunContext) -> PPOConfig:
    return build_grid_config(context, condition=CONDITION)


def run(context: RunContext):
    return run_grid_condition(context, condition=CONDITION)


__all__ = ["build_config", "run"]
