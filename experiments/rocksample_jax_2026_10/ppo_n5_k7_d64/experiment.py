"""Plain PPO baseline, pure JAX: the ``rocksample_ppo_2026_09`` recipe.

Batch 8,192 (64 envs x 128 steps), minibatch 1,024, 4 epochs (32 gradient
steps per update), clip 0.2, entropy 0.05, adaptive KL, d64 transformer.
16 seeds vmapped, 612 updates = 5,013,504 env steps each.
"""

from experiments.rocksample_jax_2026_10 import baseline, ppo

RECIPE = baseline.BaselineRecipe(
    ppo=ppo.PPOConfig(num_envs=64, num_steps=128, num_minibatches=8),
    num_seeds=16,
    updates_per_chunk=51,
    num_chunks=12,
)


def recipe(context) -> baseline.BaselineRecipe:
    return baseline.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return baseline.run(context, recipe(context))
