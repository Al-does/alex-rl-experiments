from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {"n": 4, "k": 4}
D_MODEL = 64
TOTAL_ENV_STEPS = 3_000_000

# Doubled exploration coefficient held for the first 1M steps, linearly
# annealed to 0.008 by 2M, then held through the 3M budget.
ENTROPY_COEFF_SCHEDULE = [
    [0, 0.10],
    [1_000_000, 0.10],
    [2_000_000, 0.008],
]


def build_config(context):
    return shared.build_config(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        entropy_coeff=ENTROPY_COEFF_SCHEDULE,
    )


def run(context):
    return shared.run_condition(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        total_env_steps=TOTAL_ENV_STEPS,
        entropy_coeff=ENTROPY_COEFF_SCHEDULE,
    )
