"""Control: 300M-token next-token CE on Pusher-B b10 (pure JAX)."""

from experiments.pusher_b_jax_2026_10.supervised import run_supervised

TARGET_ENV_STEPS = 300_000_000


def run(context):
    return run_supervised(
        context,
        preset="b10",
        target_env_steps=TARGET_ENV_STEPS,
        param_checkpoints=True,
    )
