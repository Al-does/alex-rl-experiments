"""Large-batch PPO baseline, pure JAX.

As ``ppo_n5_k7_d64`` but batch 65,536 (512 envs x 128 steps) split into 16
minibatches of 4,096, so each update takes 64 gradient steps (4 epochs)
instead of 32. 16 seeds vmapped, 77 updates = 5,046,272 env steps each.
"""

from experiments.rocksample_jax_2026_10 import baseline, ppo

RECIPE = baseline.BaselineRecipe(
    ppo=ppo.PPOConfig(num_envs=512, num_steps=128, num_minibatches=16),
    num_seeds=16,
    updates_per_chunk=11,
    num_chunks=7,
)


def recipe(context) -> baseline.BaselineRecipe:
    return baseline.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return baseline.run(context, recipe(context))
