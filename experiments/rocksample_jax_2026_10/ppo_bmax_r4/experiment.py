"""Round 4: the round-3 winner (``b1m_d128_loose_ep16``) at d_model 256.

Round 3 compared d256 only at 8 epochs, so width and epochs were confounded.
This runs one seed of the ep16 recipe at d256; ``num_keys=4`` keeps seed 0's
key identical to round 3's seed 0 (43.5 at 30M), giving a seed-matched
comparison.
"""

from experiments.rocksample_jax_2026_10 import ppo, sweep
from experiments.rocksample_jax_2026_10.sweep import Arm

ENV_STEPS = 30_000_000
NUM_SEEDS = 1

ARMS = (
    Arm(
        "b1m_d256_loose_ep16",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=16,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=256,
        note="r3 b1m_d128_loose_ep16 (41.3 at 30M; seed 0 43.5) at d256",
    ),
)

RECIPE = sweep.SweepRecipe(
    arms=ARMS,
    env_steps_per_seed=ENV_STEPS,
    num_seeds=NUM_SEEDS,
    num_keys=4,
)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
