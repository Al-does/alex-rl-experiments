"""Continue ``rl_b10_continue`` from its final (115.9M-step) checkpoint.

Same PPO settings as ``rl_b10_continue``; the run is bounded by wall-clock
training time (``MAX_TRAIN_TIME_S``, via Tune's ``time_total_s`` stop) rather
than by the step budget, and Tune's ``checkpoint_at_end`` saves the last
checkpoint before the box publishes and self-destructs. Launch with
``python -m experiments.pusher_b.continue_ppo``.
"""

from __future__ import annotations

import json
import os

from harness.context import RunContext

from experiments.pusher_b.rl import PPOSettings, run_ppo

PRIOR_LEAF = "rl_b10_continue"
PRIOR_RUN_ID = "20260914T050618Z-88d0fb36"
PRIOR_ENV_STEPS = 115_901_216
ADDITIONAL_ENV_STEPS = 400_000_000
MAX_TRAIN_TIME_S = 11.5 * 3600
CHECKPOINT_EVERY_ENV_STEPS = 20_000_000
TRAIN_BATCH_SIZE = 1_048_576
MINIBATCH_SIZE = 65_536
NUM_ENV_RUNNERS = 48
OVERRIDES_ENV = "PUSHER_B_CONTINUE_OVERRIDES"

SETTINGS = PPOSettings(
    total_env_steps=PRIOR_ENV_STEPS + ADDITIONAL_ENV_STEPS,
    train_batch_size=TRAIN_BATCH_SIZE,
    minibatch_size=MINIBATCH_SIZE,
    lr=1e-5,
    entropy_coeff=0.0,
    checkpoint_every_env_steps=CHECKPOINT_EVERY_ENV_STEPS,
    checkpoint_origin_env_steps=PRIOR_ENV_STEPS,
    num_env_runners=NUM_ENV_RUNNERS,
    max_train_time_s=MAX_TRAIN_TIME_S,
)


def settings() -> PPOSettings:
    """``SETTINGS`` with optional JSON field overrides from the environment."""
    raw = os.environ.get(OVERRIDES_ENV)
    if not raw:
        return SETTINGS
    overrides = json.loads(raw)
    unknown = set(overrides) - set(PPOSettings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {OVERRIDES_ENV} fields: {sorted(unknown)}")
    return PPOSettings(**{**SETTINGS.__dict__, **overrides})


def run(context: RunContext):
    if context.resume_from is None and not context.smoke:
        raise ValueError(
            "rl_b10_continue2 requires --resume-from; launch via "
            "python -m experiments.pusher_b.continue_ppo"
        )
    return run_ppo(context, preset="b10", settings=settings())
