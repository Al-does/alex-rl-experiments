"""Variant-3 REINFORCE rerun: canonical recipe, T=1.5 logits, context_len=32.

Same environment, budget, batch geometry, learning rate, and probe battery as
the 15-seed campaign; the only changes are categorical sampling temperature
``logits / 1.5`` (rollout sampling and train-time log-probability evaluation
share the module's action distribution inputs) and attention context length 32.
``grad_checkpointing`` only recomputes block activations during backward to
keep peak memory in check on ~24-48 GB cards; it does not change any math.
"""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    BASE_MODEL_CONFIG,
    build_config as _build_shared_config,
    run_condition,
)
from harness.context import RunContext

VARIANT = 3
SAMPLING_TEMPERATURE = 1.5
CONTEXT_LEN = 32

T15_CTX32_MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    "context_len": CONTEXT_LEN,
    "sampling_temperature": SAMPLING_TEMPERATURE,
    "grad_checkpointing": True,
}


def build_config(context: RunContext) -> PPOConfig:
    return _build_shared_config(
        context, VARIANT, model_config=T15_CTX32_MODEL_CONFIG
    )


def _config_builder(context: RunContext, variant: int) -> PPOConfig:
    if variant != VARIANT:
        raise ValueError(f"this experiment only supports variant {VARIANT}")
    return build_config(context)


def run(context: RunContext):
    return run_condition(
        context,
        VARIANT,
        config_builder=_config_builder,
        recipe_overrides={
            "experiment_arm": "t15_ctx32",
            "model_config": T15_CTX32_MODEL_CONFIG,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "temperature_semantics": (
                "categorical logits divided by 1.5 in rollout sampling and "
                "train-time log-probability evaluation"
            ),
            "transformer_lookback": 4 * CONTEXT_LEN,
        },
    )
