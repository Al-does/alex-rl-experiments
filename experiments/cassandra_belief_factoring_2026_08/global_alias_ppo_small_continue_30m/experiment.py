"""Continue the stable small global-alias policy for another 30M steps."""

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


SOURCE_RUN_ID = "20260820T182145Z-b4be6471"
SOURCE_STEPS = 20_054_016
ADDITIONAL_ENV_STEPS = 30_000_000
ENTROPY_COEFF = 0.008
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()


def build_config(context: RunContext) -> PPOConfig:
    """Build the 30M small global-alias continuation recipe."""

    env_config = environment_config(action_scope="global_aliases")
    env_config["initial_state_distribution"] = "all_good"
    return (
        build_shared_config(context, action_scope="global_aliases")
        .environment(
            CassandraActionObservationEnv,
            env_config=env_config,
        )
        .training(
            entropy_coeff=ENTROPY_COEFF,
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
    """Restore and train through 50M total sampled steps."""

    return continue_from_checkpoint(
        build_config(context),
        context,
        source_run_id=SOURCE_RUN_ID,
        source_steps=SOURCE_STEPS,
        additional_steps=ADDITIONAL_ENV_STEPS,
    )


__all__ = [
    "ADDITIONAL_ENV_STEPS",
    "ENTROPY_COEFF",
    "MODEL_CONFIG",
    "SOURCE_RUN_ID",
    "SOURCE_STEPS",
    "build_config",
    "run",
]
