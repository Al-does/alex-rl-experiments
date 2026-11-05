from experiments.mrrxor_grokking_jax_2026_11 import sweep


RECIPE = sweep.grid_recipe()


def recipe(context):
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
