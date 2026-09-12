"""Conservative H100-estimated PPO scaling for non-ergodic MESS3."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.nonergodic_mess3_token_guess_cycle_1.shared import (
    build_config as build_shared_config,
    resolved_recipe as resolve_shared_recipe,
    run_condition,
)
from harness.context import RunContext


CONDITION = "ppo_h100_estimated"
TRAIN_BATCH_SIZE = 262_144
MINIBATCH_SIZE = 8_192
NUM_ENVS_PER_ENV_RUNNER = 19
ASSUMED_H100_MEMORY_GB = 80.0
REFERENCE_4090_BATCH_SIZE = 32_768
REFERENCE_4090_RESERVED_MEMORY_GB = 2.22
ESTIMATED_RESERVED_MEMORY_GB = (
    REFERENCE_4090_RESERVED_MEMORY_GB
    * TRAIN_BATCH_SIZE
    / REFERENCE_4090_BATCH_SIZE
)
NAIVE_LINEAR_CAPACITY_STEPS = int(
    REFERENCE_4090_BATCH_SIZE
    * ASSUMED_H100_MEMORY_GB
    / REFERENCE_4090_RESERVED_MEMORY_GB
)


def build_config(context: RunContext) -> PPOConfig:
    return build_shared_config(
        context,
        train_batch_size=TRAIN_BATCH_SIZE,
        minibatch_size=MINIBATCH_SIZE,
        production_num_envs_per_env_runner=NUM_ENVS_PER_ENV_RUNNER,
    )


def resolved_recipe(context: RunContext) -> dict[str, object]:
    recipe = resolve_shared_recipe(
        context,
        condition=CONDITION,
        train_batch_size=TRAIN_BATCH_SIZE,
        minibatch_size=MINIBATCH_SIZE,
        production_num_envs_per_env_runner=NUM_ENVS_PER_ENV_RUNNER,
    )
    recipe["intended_hardware"] = (
        "CPU smoke; full training estimated for one NVIDIA H100 80 GB"
    )
    recipe["hardware_estimate"] = {
        "status": (
            "conservative capacity estimate validated by a short H100 NVL "
            "run; no maximum-capacity benchmark was run"
        ),
        "assumed_gpu": "NVIDIA H100 80 GB capacity estimate",
        "reference_measurement": {
            "gpu": "NVIDIA GeForce RTX 4090 24.6 GB",
            "train_batch_size_per_learner": REFERENCE_4090_BATCH_SIZE,
            "minibatch_size": REFERENCE_4090_BATCH_SIZE,
            "reserved_memory_gb": REFERENCE_4090_RESERVED_MEMORY_GB,
            "training_sequence_mode": "complete_episode",
            "attention_implementation": "sdpa",
        },
        "linear_memory_extrapolation": {
            "naive_capacity_steps_at_80_gb": NAIVE_LINEAR_CAPACITY_STEPS,
            "selected_batch_steps": TRAIN_BATCH_SIZE,
            "selected_minibatch_steps": MINIBATCH_SIZE,
            "estimated_reserved_memory_gb": ESTIMATED_RESERVED_MEMORY_GB,
            "estimated_fraction_of_80_gb": (
                ESTIMATED_RESERVED_MEMORY_GB / ASSUMED_H100_MEMORY_GB
            ),
        },
        "h100_validation_run": {
            "gpu": "NVIDIA H100 NVL 95,830 MiB",
            "sampled_steps": 2_702_560,
            "training_iterations": 10,
            "observed_reserved_memory_gb_approx": 16.0,
            "full_minibatch_optimizer_updates": 70,
            "sampling_timer_seconds_final_iteration": 103.4,
            "learner_timer_seconds_final_iteration": 3.2,
        },
        "selection_rationale": (
            "use an 8x larger PPO train batch than the measured 4090 recipe "
            "while retaining its 32,768-step minibatch; this preserves "
            "optimizer updates per sampled step because H100 wall-clock time "
            "was rollout-bound rather than learner-bound"
        ),
    }
    recipe["sampling_layout"]["configuration_basis"] = (
        "retain the measured 19 vector environments per runner; RLlib's "
        "automatic rollout fragment grows so each environment collects "
        "multiple fresh complete episodes for the larger train batch"
    )
    return recipe


def run(context: RunContext):
    return run_condition(
        context,
        condition=CONDITION,
        config_builder=build_config,
        recipe_builder=resolved_recipe,
    )
