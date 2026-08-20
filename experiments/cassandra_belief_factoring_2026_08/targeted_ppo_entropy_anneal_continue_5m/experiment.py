"""Continue the successful annealed targeted policy for five million steps."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.continuation import (
    continue_from_checkpoint,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config as build_shared_config,
    environment_config,
)
from harness.context import RunContext


SOURCE_RUN_ID = "20260820T012606Z-a674abc3"
SOURCE_STEPS = 5_013_504
ADDITIONAL_ENV_STEPS = 5_000_000
TARGET_ENV_STEPS = SOURCE_STEPS + ADDITIONAL_ENV_STEPS
ENTROPY_COEFF_SCHEDULE = [
    [0, 0.08],
    [2_500_000, 0.08],
    [5_000_000, 0.01],
]


def build_config(context: RunContext) -> PPOConfig:
    """Build the source recipe for fresh smoke validation."""

    env_config = environment_config(action_scope="targeted")
    env_config["initial_state_distribution"] = "all_good"
    return (
        build_shared_config(context, action_scope="targeted")
        .environment(
            CassandraActionObservationEnv,
            env_config=env_config,
        )
        .training(entropy_coeff=ENTROPY_COEFF_SCHEDULE)
    )


def run(context: RunContext):
    """Restore the source checkpoint and train through ten million total steps."""

    return continue_from_checkpoint(
        build_config(context),
        context,
        source_run_id=SOURCE_RUN_ID,
        source_steps=SOURCE_STEPS,
        additional_steps=ADDITIONAL_ENV_STEPS,
    )


__all__ = [
    "ADDITIONAL_ENV_STEPS",
    "SOURCE_RUN_ID",
    "SOURCE_STEPS",
    "TARGET_ENV_STEPS",
    "build_config",
    "run",
]
