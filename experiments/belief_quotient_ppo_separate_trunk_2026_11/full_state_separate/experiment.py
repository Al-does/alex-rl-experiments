from experiments.belief_quotient_ppo_separate_trunk_2026_11.shared import run_condition


def run(context):
    return run_condition(context, variant=3, arm="separate")
