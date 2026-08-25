"""Continue completed targeted 250M agents for another 250M steps."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.continuation_shared import (
    CONTINUE_LIFETIME_ENV_STEPS,
    build_continue_config,
    run_continuation,
)
from harness.context import RunContext


CONDITION = "best_critic_bptt64_targeted_continue_250m"
HYPOTHESIS = (
    "Resume the completed targeted best-critic BPTT-64 agents and train "
    "another 250M environment steps with unchanged entropy."
)


def build_config(context: RunContext) -> PPOConfig:
    return build_continue_config(context)


def run(context: RunContext):
    return run_continuation(
        context,
        condition=CONDITION,
        hypothesis=HYPOTHESIS,
        lifetime_env_steps=CONTINUE_LIFETIME_ENV_STEPS,
        config_builder=build_continue_config,
    )


__all__ = ["CONDITION", "HYPOTHESIS", "build_config", "run"]
