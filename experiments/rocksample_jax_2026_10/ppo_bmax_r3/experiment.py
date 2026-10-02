"""Round 3: the three best round-2 arms extended from 10M to 30M env steps.

Every batch-1M arm was still climbing 2-4 return per update at the end of its
10 updates in ``ppo_bmax_r2``. Round 2 saved no trainer state, so these are
fresh 30-update runs (same seed keys as round 2, so the first 10 updates
reproduce round 2 up to GPU nondeterminism). This round saves the final
trainer state per seed in ``artifacts/<arm>/seed<i>.pkl`` so a later leaf can
continue via ``Arm.resume_from`` instead of retraining.
"""

from experiments.rocksample_jax_2026_10 import ppo, sweep
from experiments.rocksample_jax_2026_10.sweep import Arm

ENV_STEPS = 30_000_000
NUM_SEEDS = 4

ARMS = (
    Arm(
        "b1m_d128_loose_ep16",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=16,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=128,
        note="r2 b1m_d128_loose_ep16 (29.2 at 10M) to 30M",
    ),
    Arm(
        "b1m_d128_push",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=16,
            lr=1e-3, clip_param=0.5, kl_target=0.1, entropy_coeff=0.01,
        ),
        d_model=128,
        note="r2 b1m_d128_push (29.9 at 10M) to 30M",
    ),
    Arm(
        "b1m_d256_loose",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=256,
        note="r2 b1m_d256_loose (28.2 at 10M) to 30M",
    ),
)

RECIPE = sweep.SweepRecipe(
    arms=ARMS,
    env_steps_per_seed=ENV_STEPS,
    num_seeds=NUM_SEEDS,
)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
