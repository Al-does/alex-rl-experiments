from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {}
D_MODEL = 128
TOTAL_ENV_STEPS = 5_000_000


def build_config(context):
    return shared.build_config(context, env_config=ENV_CONFIG, d_model=D_MODEL)


def run(context):
    return shared.run_condition(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        total_env_steps=TOTAL_ENV_STEPS,
    )
