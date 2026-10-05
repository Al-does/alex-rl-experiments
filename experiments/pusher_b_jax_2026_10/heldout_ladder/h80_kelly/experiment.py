"""Held-out ladder rung: 80% of the Pusher-B b10 sequence pool held out, next-token CE + decoupled Kelly wager head, 400M env steps."""

from experiments.pusher_b_jax_2026_10.heldout import run_heldout


def run(context):
    return run_heldout(context, heldout_fraction=0.80, kelly=True)
