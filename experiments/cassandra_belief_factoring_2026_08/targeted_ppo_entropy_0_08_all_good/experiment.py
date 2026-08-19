"""Train targeted Cassandra PPO with moderate entropy and all-good starts."""

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    SMOKE_ENV_STEPS,
    TOTAL_ENV_STEPS,
    build_config as build_shared_config,
    environment_config,
)
from harness.context import RunContext
from harness.runners import run_tune


ENTROPY_COEFF = 0.08


def build_config(context: RunContext) -> PPOConfig:
    """Build the moderate-entropy, all-good targeted PPO recipe."""

    env_config = environment_config(action_scope="targeted")
    env_config["initial_state_distribution"] = "all_good"
    return (
        build_shared_config(context, action_scope="targeted")
        .environment(
            CassandraActionObservationEnv,
            env_config=env_config,
        )
        .training(entropy_coeff=ENTROPY_COEFF)
    )


def run(context: RunContext):
    """Train for the requested budget without checkpoint probing."""

    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    return run_tune(
        build_config(context),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )


__all__ = ["ENTROPY_COEFF", "build_config", "run"]
