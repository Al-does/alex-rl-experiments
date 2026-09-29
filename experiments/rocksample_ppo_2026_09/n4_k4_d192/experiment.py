from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {"n": 4, "k": 4}
D_MODEL = 192


def build_config(context):
    return shared.build_config(context, env_config=ENV_CONFIG, d_model=D_MODEL)


def run(context):
    return shared.run_condition(context, env_config=ENV_CONFIG, d_model=D_MODEL)
