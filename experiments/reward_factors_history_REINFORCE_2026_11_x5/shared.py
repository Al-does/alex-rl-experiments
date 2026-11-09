"""5x-budget continuation of ``reward_factors_history_REINFORCE_2026_11``.

Each leaf restores the parent leaf's completed seed-matched final checkpoint
(fetched from B2 via the parent's checked-in ``run_manifest.json``) and keeps
training to five times the parent budget with the parent's configuration.
"""

from pathlib import Path

from experiments.belief_quotient_history_REINFORCE_2026_11.continuation import prepare_continuation
from experiments.belief_quotient_history_REINFORCE_2026_11.training import run_study
from experiments.reward_factors_history_REINFORCE_2026_11 import shared as parent
from experiments.reward_factors_history_REINFORCE_2026_11.analysis import analyze_checkpoint


STUDY = "reward_factors_history_REINFORCE_2026_11_x5"
BUDGET_MULTIPLIER = 5
TOTAL_ENV_STEPS = parent.TOTAL_ENV_STEPS * BUDGET_MULTIPLIER
PARENT_ROOT = Path(parent.__file__).resolve().parent


def build_config(context, condition, *, architecture):
    return parent.build_config(context, condition, architecture=architecture)


def run_condition(context, condition, *, architecture):
    source = prepare_continuation(
        context, parent_leaf_dir=PARENT_ROOT / f"{condition}_{architecture}"
    )
    recipe = {
        **parent.condition_recipe(condition, architecture=architecture),
        "study": STUDY,
        "parent_study": parent.STUDY,
        "budget_multiplier": BUDGET_MULTIPLIER,
        "total_env_steps": TOTAL_ENV_STEPS,
        "continuation": (
            "restores the parent's final Tune checkpoint (weights, optimizer, lifetime "
            "counters, iteration); no entropy bonus, as in the parent; environment RNG "
            "streams restart from the seed"
        ),
    }
    return run_study(
        context,
        build_config(context, condition, architecture=architecture),
        recipe=recipe,
        analyze_checkpoint=analyze_checkpoint,
        continue_from=source,
    )
