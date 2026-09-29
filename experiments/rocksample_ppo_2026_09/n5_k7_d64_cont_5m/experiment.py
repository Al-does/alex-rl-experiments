"""Continue the width-64 RockSample[5,7] PPO run for another 5M steps.

Restores the final algorithm checkpoint of ``n5_k7_d64`` run
``20260929T205026Z-e05f80e0`` (the +20.6 arm) passed via
``--resume-from``, then trains for an additional 5M env steps past the
restored lifetime count. Optimizer state, module weights, and env-runner
counters carry over; all other hyperparameters match the source run.
"""

from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {}
D_MODEL = 64
ADDITIONAL_ENV_STEPS = 5_000_000
SOURCE_RUN_ID = "20260929T205026Z-e05f80e0"


def build_config(context):
    return shared.build_config(context, env_config=ENV_CONFIG, d_model=D_MODEL)


def run(context):
    if context.resume_from is None:
        raise ValueError(
            "continuation leaf requires --resume-from <algorithm "
            f"checkpoint of {SOURCE_RUN_ID}>"
        )
    return shared.run_continuation(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        additional_env_steps=ADDITIONAL_ENV_STEPS,
    )
