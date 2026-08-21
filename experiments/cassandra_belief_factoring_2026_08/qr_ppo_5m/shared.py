"""Matched fixed-quantile PPO recipes for Cassandra maintenance."""

from __future__ import annotations

from typing import Any

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
from learners import QRPPOTorchLearner
from learners.models import (
    QRValueMixin,
    TransformerModel,
    TransformerModelConfig,
)


TOTAL_ENV_STEPS = 5_000_000
ENTROPY_COEFF = 0.008
GAMMA = 0.990
NUM_QUANTILES = 64
QUANTILE_HUBER_KAPPA = 10.0
QUANTILE_LOSS_COEFFICIENT = 0.5
MODEL_CONFIG: dict[str, Any] = {
    **TransformerModelConfig(
        d_model=64,
        n_layers=4,
        n_heads=1,
        context_len=256,
        max_seq_len=256,
    ).to_dict(),
    "qr_value": {"num_quantiles": NUM_QUANTILES},
}
_ACTION_SCOPES = {"global_aliases", "targeted"}


class CassandraQRTransformer(QRValueMixin, TransformerModel):
    """Causal Cassandra policy with a fixed-quantile value critic."""


def build_config(context: RunContext, *, action_scope: str) -> PPOConfig:
    """Build one matched QR-PPO Cassandra condition."""

    if action_scope not in _ACTION_SCOPES:
        raise ValueError(
            "QR Cassandra action_scope must be 'global_aliases' or 'targeted'"
        )
    env_config = environment_config(action_scope=action_scope)
    env_config["initial_state_distribution"] = "all_good"
    return (
        build_shared_config(context, action_scope=action_scope)
        .environment(
            CassandraActionObservationEnv,
            env_config=env_config,
        )
        .training(
            entropy_coeff=ENTROPY_COEFF,
            gamma=GAMMA,
            use_kl_loss=False,
            kl_coeff=0.0,
            # QR is the sole critic objective. The quantile mean still supplies
            # PPO's scalar baseline for GAE and bootstrapping.
            vf_loss_coeff=0.0,
        )
        .learners(
            learner_class=QRPPOTorchLearner,
            learner_config_dict={
                "qr_value/loss_coefficient": QUANTILE_LOSS_COEFFICIENT,
                "qr_value/huber_kappa": QUANTILE_HUBER_KAPPA,
            },
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CassandraQRTransformer,
                model_config=dict(MODEL_CONFIG),
            )
        )
    )


def run_condition(context: RunContext, *, action_scope: str):
    """Train one QR-PPO condition for the matched budget."""

    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    return run_tune(
        build_config(context, action_scope=action_scope),
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
    "CassandraQRTransformer",
    "ENTROPY_COEFF",
    "GAMMA",
    "MODEL_CONFIG",
    "NUM_QUANTILES",
    "QUANTILE_HUBER_KAPPA",
    "QUANTILE_LOSS_COEFFICIENT",
    "TOTAL_ENV_STEPS",
    "build_config",
    "run_condition",
]
