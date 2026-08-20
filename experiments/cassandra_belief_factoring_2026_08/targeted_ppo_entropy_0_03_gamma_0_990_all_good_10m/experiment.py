"""Train targeted PPO for 10M steps with moderate constant entropy."""

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    SMOKE_ENV_STEPS,
    build_config as build_shared_config,
    environment_config,
)
from harness.context import RunContext
from harness.runners import run_tune


TOTAL_ENV_STEPS = 10_000_000
ENTROPY_COEFF = 0.03
GAMMA = 0.990


def build_config(context: RunContext) -> PPOConfig:
    """Build the standard-transformer 10M targeted PPO recipe."""

    env_config = environment_config(action_scope="targeted")
    env_config["initial_state_distribution"] = "all_good"
    return (
        build_shared_config(context, action_scope="targeted")
        .environment(
            CassandraActionObservationEnv,
            env_config=env_config,
        )
        .training(
            entropy_coeff=ENTROPY_COEFF,
            gamma=GAMMA,
            use_kl_loss=False,
            kl_coeff=0.0,
        )
    )


def run(context: RunContext):
    """Train without longitudinal probing."""

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


__all__ = [
    "ENTROPY_COEFF",
    "GAMMA",
    "TOTAL_ENV_STEPS",
    "build_config",
    "run",
]
