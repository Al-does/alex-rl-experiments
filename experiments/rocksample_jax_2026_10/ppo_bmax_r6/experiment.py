"""Round 6: few-seed confirmation of the round-5 arms that matched the control.

Round 5 (seed 0, 30M) found returns quantised at ~34 / ~39 / ~43.5; only
``d128_kl0.1`` (43.6) and ``d128_mb8k`` (43.7) reached the control's basin
(``b1m_d128_loose_ep16`` seed 0, 43.5; its 4 seeds were 43.5 / 34.1 / 43.4 /
44.1). This runs seeds 1-3 of both, plus their combination, with the same keys
as the control (``num_keys=4``), so each arm gets 4 seed-matched seeds and is
scored by how many land in the ~43.5 basin.
"""

from dataclasses import replace

from experiments.rocksample_jax_2026_10 import ppo, sweep
from experiments.rocksample_jax_2026_10.sweep import Arm

ENV_STEPS = 30_000_000
NUM_SEEDS = 3
FIRST_SEED = 1

BASE = ppo.PPOConfig(
    num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=16,
    lr=1e-3, clip_param=0.3, kl_target=0.05,
)

ARMS = (
    Arm("d128_kl0.1", replace(BASE, kl_target=0.1), d_model=128, note="r5 seed 0: 43.6"),
    Arm("d128_mb8k", replace(BASE, num_minibatches=128), d_model=128, note="r5 seed 0: 43.7"),
    Arm(
        "d128_kl0.1_mb8k",
        replace(BASE, kl_target=0.1, num_minibatches=128),
        d_model=128,
        note="both r5 matches combined (not run at seed 0)",
    ),
)

RECIPE = sweep.SweepRecipe(
    arms=ARMS,
    env_steps_per_seed=ENV_STEPS,
    num_seeds=NUM_SEEDS,
    num_keys=4,
    first_seed=FIRST_SEED,
)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
