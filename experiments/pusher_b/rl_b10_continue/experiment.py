"""Continue ``rl_b10`` from its final checkpoint with a larger PPO batch.

Launch with ``python -m experiments.pusher_b.continue_ppo`` so the prior run's
checkpoint is restored from B2 and passed as ``resume_from``. Optimizer
schedules are held at their annealed values (lr 1e-5, entropy 0), matching the
tail of the original 15M-step schedule.
"""

from __future__ import annotations

import json
import os

from harness.context import RunContext

from experiments.pusher_b.rl import PPOSettings, run_ppo

PRIOR_LEAF = "rl_b10"
PRIOR_RUN_ID = "20260913T221640Z-e0010176"
PRIOR_ENV_STEPS = 15_134_336
ADDITIONAL_ENV_STEPS = 100_000_000
CHECKPOINT_EVERY_ENV_STEPS = 20_000_000
TRAIN_BATCH_SIZE = 1_048_576
MINIBATCH_SIZE = 65_536
OVERRIDES_ENV = "PUSHER_B_CONTINUE_OVERRIDES"

SETTINGS = PPOSettings(
    total_env_steps=PRIOR_ENV_STEPS + ADDITIONAL_ENV_STEPS,
    train_batch_size=TRAIN_BATCH_SIZE,
    minibatch_size=MINIBATCH_SIZE,
    lr=1e-5,
    entropy_coeff=0.0,
    checkpoint_every_env_steps=CHECKPOINT_EVERY_ENV_STEPS,
    checkpoint_origin_env_steps=PRIOR_ENV_STEPS,
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
            "rl_b10_continue requires --resume-from; launch via "
            "python -m experiments.pusher_b.continue_ppo"
        )
    return run_ppo(context, preset="b10", settings=settings())
