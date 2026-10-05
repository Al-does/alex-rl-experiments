"""Short legacy PyTorch supervised-CE run (b10) for the JAX comparison.

Needs ``experiments/pusher_b`` (branch ``devin/1789433175-pusher-b-ntce``).
Same recipe as ``pusher_b/supervised_b10`` but stops after 3,000 updates.
"""

from functools import partial

TOTAL_STEPS = 3_000
CHECKPOINT_STEPS = (1, 3, 10, 30, 100, 300, 1_000, 3_000)


def run(context):
    from experiments.pusher_b import supervised

    if not context.smoke:
        supervised.TrainingConfig = partial(
            supervised.TrainingConfig,
            total_steps=TOTAL_STEPS,
            checkpoint_steps=CHECKPOINT_STEPS,
        )
    return supervised.run_supervised(context, preset="b10")
