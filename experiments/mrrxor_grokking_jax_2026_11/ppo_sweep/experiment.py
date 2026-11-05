from experiments.mrrxor_grokking_jax_2026_11 import sweep


RECIPE = sweep.grid_recipe()


def recipe(context):
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    if context.artifacts_dir != context.results_dir / "artifacts":
        raise ValueError("Use python -m experiments.mrrxor_grokking_jax_2026_11.sweep "
                         "to resolve a single run folder before harness metadata is written")
    return sweep.run(context, recipe(context))
