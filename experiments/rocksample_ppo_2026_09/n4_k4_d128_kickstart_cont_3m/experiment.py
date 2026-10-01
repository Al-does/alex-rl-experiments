"""Continue the kickstarted d128 RockSample[4,4] student for +3M steps.

Restores the final algorithm checkpoint of ``n4_k4_d128_kickstart_3m``
run ``20260929T221016Z-eecd0ffa`` (kickstart phase complete, student at
+10) passed via ``--resume-from``. The kickstart coefficient is a
constant 0.0 here — the anneal already finished, so this is pure PPO and
no teacher checkpoint is needed. The same learner class is kept so the
restored learner state maps one-to-one.
"""

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from experiments.rocksample_ppo_2026_09 import shared
from losses.kickstart import KickstartTeacherKLMixin


ENV_CONFIG = {"n": 4, "k": 4}
D_MODEL = 128
ADDITIONAL_ENV_STEPS = 3_000_000
SOURCE_RUN_ID = "20260929T221016Z-eecd0ffa"


class KickstartPPOLearner(KickstartTeacherKLMixin, PPOTorchLearner):
    pass


def build_config(context):
    return shared.build_config(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        learner_class=KickstartPPOLearner,
        learner_config_dict={"kickstart/coeff": [[0, 0.0]]},
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
        learner_class=KickstartPPOLearner,
        learner_config_dict={"kickstart/coeff": [[0, 0.0]]},
    )
