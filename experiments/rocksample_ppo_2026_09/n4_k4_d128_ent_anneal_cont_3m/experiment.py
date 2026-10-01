"""Continue the d128 ent-anneal RockSample[4,4] run for another 3M steps.

Restores the final algorithm checkpoint of ``n4_k4_d128_ent_anneal_3m``
run ``20260929T221006Z-856495e7`` (the +24.4 arm) passed via
``--resume-from``. Entropy stays on the source schedule, which evaluates
to the terminal 0.008 level throughout the continuation.
"""

from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {"n": 4, "k": 4}
D_MODEL = 128
ADDITIONAL_ENV_STEPS = 3_000_000
SOURCE_RUN_ID = "20260929T221006Z-856495e7"
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
        entropy_coeff=ENTROPY_COEFF_SCHEDULE,
    )
