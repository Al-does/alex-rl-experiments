"""Continue the successful annealed targeted policy for five million steps."""

from collections.abc import Mapping
from numbers import Real
from typing import Any

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    SMOKE_ENV_STEPS,
    build_config as build_shared_config,
    environment_config,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_algorithm


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


def _sampled_steps(result: Mapping[str, Any]) -> int:
    runners = result.get("env_runners", {})
    value = (
        runners.get("num_env_steps_sampled_lifetime")
        if isinstance(runners, Mapping)
        else None
    )
    if not isinstance(value, Real):
        value = result.get("num_env_steps_sampled_lifetime")
    if not isinstance(value, Real):
        raise KeyError("result has no lifetime sampled-step metric")
    return int(value)


def run(context: RunContext):
    """Restore the source checkpoint and train through ten million total steps."""

    if not context.smoke and context.resume_from is None:
        raise ValueError(
            "full continuation requires --resume-from with the source "
            "Algorithm checkpoint"
        )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TARGET_ENV_STEPS
    result = run_algorithm(
        build_config(context),
        context,
        should_stop=lambda metrics: _sampled_steps(metrics) >= target_steps,
        checkpoint_at_end=True,
    )
    RunArtifacts.from_context(context).write_json(
        "continuation_summary.json",
        {
            "source_run_id": SOURCE_RUN_ID,
            "source_checkpoint": context.resume_from,
            "source_steps": 0 if context.smoke else SOURCE_STEPS,
            "additional_requested_steps": (
                SMOKE_ENV_STEPS if context.smoke else ADDITIONAL_ENV_STEPS
            ),
            "final_sampled_steps": _sampled_steps(result),
        },
    )
    return result


__all__ = [
    "ADDITIONAL_ENV_STEPS",
    "SOURCE_RUN_ID",
    "SOURCE_STEPS",
    "TARGET_ENV_STEPS",
    "build_config",
    "run",
]
