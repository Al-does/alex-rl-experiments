"""Experimental arm: 300M-token next-token CE plus a decoupled Kelly wager
head on Pusher-B b10 (pure JAX).

Same trunk, optimizer, data stream, and budget as ``supervised_b10_300m``;
adds a two-logit head whose per-class wager bets on the sampled next-token
guess being right — the ``decoupled_kelly`` objective of
``experiments/mess3_token_guess_cycle_2`` ported off PPO.
"""

from experiments.pusher_b_jax_2026_10.supervised import run_supervised

TARGET_ENV_STEPS = 300_000_000


def run(context):
    return run_supervised(
        context,
        preset="b10",
        kelly=True,
        target_env_steps=TARGET_ENV_STEPS,
        param_checkpoints=True,
    )
