"""Train a narrow four-layer targeted transformer for 10M steps."""

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

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
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 10_000_000
ENTROPY_COEFF = 0.03
GAMMA = 0.990
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()


def build_config(context: RunContext) -> PPOConfig:
    """Build the narrow four-layer 10M targeted PPO recipe."""

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
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=MODEL_CONFIG,
            )
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
    "MODEL_CONFIG",
    "TOTAL_ENV_STEPS",
    "build_config",
    "run",
]
