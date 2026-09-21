"""Variant-1 REINFORCE rerun: canonical recipe, context_len=32, entropy anneal.

Same environment, budget, batch geometry, learning rate, and probe battery as
the 15-seed campaign. Changes: attention context length 32, canonical
``T=1.0`` sampling temperature (``logits / T`` wiring kept, no-op), and an
entropy bonus of 0.03 that anneals linearly to 0.0 on env-steps-sampled,
reaching zero when 1M steps of the run's step budget remain.
``grad_checkpointing`` only recomputes block activations during backward to
keep peak memory in check on ~24-48 GB cards; it does not change any math.
"""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    BASE_MODEL_CONFIG,
    _resolve_step_target,
    build_config as _build_shared_config,
    run_condition,
)
from harness.context import RunContext

VARIANT = 1
SAMPLING_TEMPERATURE = 1.0
CONTEXT_LEN = 32
ENTROPY_COEFF = 0.03
# The bonus reaches 0.0 when this many env steps remain in the run budget.
ENTROPY_ANNEAL_REMAINING_STEPS = 1_000_000

CTX32_ENT_MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    "context_len": CONTEXT_LEN,
    "sampling_temperature": SAMPLING_TEMPERATURE,
    "grad_checkpointing": True,
}


def entropy_schedule(context: RunContext) -> list[list[float]]:
    anneal_at = max(
        1, _resolve_step_target(context) - ENTROPY_ANNEAL_REMAINING_STEPS
    )
    return [[0, ENTROPY_COEFF], [anneal_at, 0.0]]


def build_config(context: RunContext) -> PPOConfig:
    config = _build_shared_config(
        context, VARIANT, model_config=CTX32_ENT_MODEL_CONFIG
    )
    return config.training(entropy_coeff=entropy_schedule(context))


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
            "experiment_arm": "ctx32_ent",
            "model_config": CTX32_ENT_MODEL_CONFIG,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "temperature_semantics": (
                "categorical logits divided by 1.0 in rollout sampling and "
                "train-time log-probability evaluation (canonical temperature)"
            ),
            "entropy_coeff_schedule": entropy_schedule(context),
            "entropy_anneal_remaining_steps": ENTROPY_ANNEAL_REMAINING_STEPS,
            "transformer_lookback": 4 * CONTEXT_LEN,
        },
    )
