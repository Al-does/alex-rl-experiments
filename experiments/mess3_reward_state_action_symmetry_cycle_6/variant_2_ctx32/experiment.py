"""Variant-2 REINFORCE rerun: canonical recipe, context_len=32, T=1.0.

Same environment, budget, batch geometry, learning rate, and probe battery as
the 15-seed campaign; the only change is attention context length 32. Sampling
temperature stays at the canonical ``T=1.0`` (``logits / T`` is still routed
through the module's action distribution inputs, a no-op at T=1.0).
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

VARIANT = 2
SAMPLING_TEMPERATURE = 1.0
CONTEXT_LEN = 32

CTX32_MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    "context_len": CONTEXT_LEN,
    "sampling_temperature": SAMPLING_TEMPERATURE,
    "grad_checkpointing": True,
}


def build_config(context: RunContext) -> PPOConfig:
    return _build_shared_config(
        context, VARIANT, model_config=CTX32_MODEL_CONFIG
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
            "experiment_arm": "ctx32",
            "model_config": CTX32_MODEL_CONFIG,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "temperature_semantics": (
                "categorical logits divided by 1.0 in rollout sampling and "
                "train-time log-probability evaluation (canonical temperature)"
            ),
            "transformer_lookback": 4 * CONTEXT_LEN,
        },
    )
