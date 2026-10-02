"""Round 6b: seeds 2-3 of round 6's ``d128_kl0.1_mb8k`` arm.

Round 6 ran out of disk while saving this arm's seed-2 trainer state; seeds 1-3
of the other two arms and seed 1 here (38.9) had finished. Same keys
(``num_keys=4``), so the four seeds stay matched to the control.
"""

from experiments.rocksample_jax_2026_10 import sweep
from experiments.rocksample_jax_2026_10.ppo_bmax_r6 import experiment as r6

(COMBO,) = [arm for arm in r6.ARMS if arm.name == "d128_kl0.1_mb8k"]

RECIPE = sweep.SweepRecipe(
    arms=(COMBO,),
    env_steps_per_seed=r6.ENV_STEPS,
    num_seeds=2,
    num_keys=4,
    first_seed=2,
)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
