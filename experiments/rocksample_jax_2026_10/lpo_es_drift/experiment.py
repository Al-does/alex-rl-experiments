"""ES meta-learning of an LPO drift objective on RockSample[5,7], pure JAX.

Port of ``rocksample_meta_rl_2026_09/lpo_es_drift`` off RLlib. Outer loop:
antithetic OpenAI-ES over the drift network, the PPO-drift weight and the log
entropy coefficient, starting exactly at PPO (clip 0.2, entropy 0.05). Inner
loop: a fresh d64 transformer trained from scratch with the candidate
objective; the whole population trains as one vmapped JAX program. Fitness is
the mean undiscounted return over the final 20% of updates. Requires the
``jax-cuda`` (GPU) or ``jax`` (CPU) dependency group. ``--resume-from``
expects a ``meta_state.json`` written by this recipe.
"""

from experiments.rocksample_jax_2026_10 import meta, ppo
from experiments.rocksample_jax_2026_10.es import OpenES
from experiments.rocksample_jax_2026_10.lpo import DriftSpec

INNER_ENV_STEPS = 3_000_000
SMOKE_INNER_ENV_STEPS = 2_048
NUM_PAIRS = 8
SMOKE_NUM_PAIRS = 2
GENERATIONS = 50
SMOKE_GENERATIONS = 2
DRIFT = DriftSpec(hidden=32, clip_param=0.2, initial_entropy_coeff=0.05)


def inner_recipe(context) -> meta.InnerRecipe:
    return meta.InnerRecipe(
        ppo=ppo.PPOConfig(
            num_envs=8 if context.smoke else 64,
            num_steps=128,
            num_minibatches=2 if context.smoke else 8,
            clip_param=DRIFT.clip_param,
            entropy_coeff=DRIFT.initial_entropy_coeff,
        ),
        total_env_steps=SMOKE_INNER_ENV_STEPS if context.smoke else INNER_ENV_STEPS,
        d_model=64,
    )


def meta_recipe(context) -> meta.MetaRecipe:
    return meta.MetaRecipe(
        drift=DRIFT,
        es=OpenES(
            num_pairs=SMOKE_NUM_PAIRS if context.smoke else NUM_PAIRS,
            sigma=0.05,
            learning_rate=0.02,
        ),
        generations=SMOKE_GENERATIONS if context.smoke else GENERATIONS,
        fitness_tail_fraction=0.2,
    )


def run(context):
    return meta.run_meta_training(context, inner_recipe(context), meta_recipe(context))
