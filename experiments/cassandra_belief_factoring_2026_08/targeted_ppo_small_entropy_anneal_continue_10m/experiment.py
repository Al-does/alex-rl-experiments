"""Continue the small targeted policy while annealing entropy to 0.008."""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

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
from learners.models.transformer import TransformerModel, TransformerModelConfig


SOURCE_RUN_ID = "20260820T060017Z-3d46dd5a"
SOURCE_STEPS = 10_027_008
ADDITIONAL_ENV_STEPS = 10_000_000
ANNEAL_END_STEPS = SOURCE_STEPS + 2_500_000
ENTROPY_COEFF_SCHEDULE = [
    [0, 0.03],
    [SOURCE_STEPS, 0.03],
    [ANNEAL_END_STEPS, 0.008],
]
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()


def build_config(context: RunContext) -> PPOConfig:
    """Build the continued small targeted recipe."""

    env_config = environment_config(action_scope="targeted")
    env_config["initial_state_distribution"] = "all_good"
    return (
        build_shared_config(context, action_scope="targeted")
        .environment(
            CassandraActionObservationEnv,
            env_config=env_config,
        )
        .training(
            entropy_coeff=ENTROPY_COEFF_SCHEDULE,
            gamma=0.990,
            use_kl_loss=False,
            kl_coeff=0.0,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=MODEL_CONFIG,
            )
        )
    )


def run(context: RunContext):
    """Restore and train through 20M total sampled steps."""

    return continue_from_checkpoint(
        build_config(context),
        context,
        source_run_id=SOURCE_RUN_ID,
        source_steps=SOURCE_STEPS,
        additional_steps=ADDITIONAL_ENV_STEPS,
    )


__all__ = [
    "ADDITIONAL_ENV_STEPS",
    "ANNEAL_END_STEPS",
    "ENTROPY_COEFF_SCHEDULE",
    "MODEL_CONFIG",
    "SOURCE_RUN_ID",
    "SOURCE_STEPS",
    "build_config",
    "run",
]
