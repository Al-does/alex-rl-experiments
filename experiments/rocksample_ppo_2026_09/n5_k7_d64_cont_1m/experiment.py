"""Advance the d64 RockSample[5,7] run from 1M to 2M env steps.

Resumes the same ``n5_k7_d64`` 1M checkpoint used by
``n5_k7_d64_cont_5m`` (``--resume-from``) and trains +1M, producing a
final algorithm checkpoint at ~2M lifetime steps — the
``default_policy`` RLModule inside it is the teacher for
``n5_k7_d128_kickstart_5m``.
"""

from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {}
D_MODEL = 64
ADDITIONAL_ENV_STEPS = 1_000_000
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
