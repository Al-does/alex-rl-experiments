"""5x-budget continuation of ``belief_quotient_history_REINFORCE_2026_11``.

Each leaf restores the parent leaf's completed seed-matched final checkpoint
(fetched from B2 via the parent's checked-in ``run_manifest.json``) and keeps
training to five times the parent budget. Configuration is the parent's:
the entropy schedule was already annealed to zero at 9M steps, so the
continuation trains with entropy 0.
"""

from pathlib import Path

from experiments.belief_quotient_history_REINFORCE_2026_11 import shared as parent
from experiments.belief_quotient_history_REINFORCE_2026_11.analysis import analyze_checkpoint
from experiments.belief_quotient_history_REINFORCE_2026_11.continuation import prepare_continuation
from experiments.belief_quotient_history_REINFORCE_2026_11.training import run_study


STUDY = "belief_quotient_history_REINFORCE_2026_11_x5"
BUDGET_MULTIPLIER = 5
TOTAL_ENV_STEPS = parent.TOTAL_ENV_STEPS * BUDGET_MULTIPLIER
PARENT_ROOT = Path(parent.__file__).resolve().parent


def build_config(context, variant, *, architecture):
    return parent.build_config(context, variant, architecture=architecture)


def run_condition(context, variant, *, architecture):
    source = prepare_continuation(
        context, parent_leaf_dir=PARENT_ROOT / f"variant_{variant}_{architecture}"
    )
    recipe = {
        **parent.condition_recipe(variant, architecture=architecture),
        "study": STUDY,
        "parent_study": parent.STUDY,
        "budget_multiplier": BUDGET_MULTIPLIER,
        "total_env_steps": TOTAL_ENV_STEPS,
        "continuation": (
            "restores the parent's final Tune checkpoint (weights, optimizer, lifetime "
            "counters, iteration); parent entropy schedule reached 0 at 9M steps, so the "
            "continuation uses entropy 0; environment RNG streams restart from the seed"
        ),
    }
    return run_study(
        context,
        build_config(context, variant, architecture=architecture),
        recipe=recipe,
        analyze_checkpoint=analyze_checkpoint,
        continue_from=source,
    )
