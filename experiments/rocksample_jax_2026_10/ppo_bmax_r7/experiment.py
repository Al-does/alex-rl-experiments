"""Round 7: ``d128_kl0.1`` again, all 4 matched seeds, with 10 log-spaced checkpoints.

Rounds 5/6 kept only each seed's final trainer state (43.6 / 39.1 / 43.7 /
43.7). This reruns the same arm with the same keys (``num_keys=4``) and saves
rollout-free checkpoints after updates 1, 2, 3, 4, 5, 6, 9, 14, 20 and the
full final state after update 29 (~30.4M steps).
"""

from experiments.rocksample_jax_2026_10 import sweep
from experiments.rocksample_jax_2026_10.ppo_bmax_r6 import experiment as r6

(ARM,) = [arm for arm in r6.ARMS if arm.name == "d128_kl0.1"]

RECIPE = sweep.SweepRecipe(
    arms=(ARM,),
    env_steps_per_seed=r6.ENV_STEPS,
    num_seeds=4,
    num_keys=4,
    num_checkpoints=10,
)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
