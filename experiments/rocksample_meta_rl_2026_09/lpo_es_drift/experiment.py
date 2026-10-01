"""ES meta-learning of an LPO drift objective on RockSample[5,7].

Outer loop: antithetic OpenAI-ES over the drift network, the PPO-drift weight
and the log entropy coefficient, starting exactly at PPO (clip 0.2, entropy
0.05). Inner loop: a fresh d64 transformer trained from scratch with the
candidate objective; fitness is the mean undiscounted return over the final
20% of iterations. Antithetic pairs share an inner seed. ``--resume-from``
expects a ``meta_state.json`` written by this recipe.
"""

from experiments.rocksample_meta_rl_2026_09 import shared
from experiments.rocksample_meta_rl_2026_09.es import OpenES
from experiments.rocksample_meta_rl_2026_09.learned_objective import DriftSpec

ENV_CONFIG = {}
D_MODEL = 64
INNER_ENV_STEPS = 3_000_000
SMOKE_INNER_ENV_STEPS = 2_048
NUM_PAIRS = 8
SMOKE_NUM_PAIRS = 2
GENERATIONS = 50
SMOKE_GENERATIONS = 2
DRIFT = DriftSpec(hidden=32, clip_param=0.2, initial_entropy_coeff=0.05)


def inner_recipe(context) -> shared.InnerRecipe:
    return shared.InnerRecipe(
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        total_env_steps=SMOKE_INNER_ENV_STEPS if context.smoke else INNER_ENV_STEPS,
        train_batch_size=1_024 if context.smoke else 8_192,
        minibatch_size=256 if context.smoke else 1_024,
        num_env_runners=4,
        gpus_per_trial=0.25,
    )


def meta_recipe(context) -> shared.MetaRecipe:
    return shared.MetaRecipe(
        drift=DRIFT,
        es=OpenES(
            num_pairs=SMOKE_NUM_PAIRS if context.smoke else NUM_PAIRS,
            sigma=0.05,
            learning_rate=0.02,
        ),
        generations=SMOKE_GENERATIONS if context.smoke else GENERATIONS,
        fitness_tail_fraction=0.2,
    )


def build_config(context):
    recipe = meta_recipe(context)
    params = recipe.drift.initial_params(context.seed).tolist()
    candidate = {"index": 0, "role": "center", "seed": context.seed, "params": params}
    return shared.build_inner_config(
        context, inner_recipe(context), [candidate], recipe.drift
    )


def run(context):
    return shared.run_meta_training(
        context, inner_recipe(context), meta_recipe(context)
    )
