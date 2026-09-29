"""Width-128 student on RockSample[5,7] kickstarted by the 2M d64 teacher.

The teacher is the ``default_policy`` RLModule from the final checkpoint
of the d64 [5,7] continuation at ~2M lifetime steps (produced by
``n5_k7_d64_cont_1m``), staged at ``artifacts/teacher_n5_k7_d64_2m/``
before launch. Same anneal as the [4,4] kickstart arms: forward KL on
policy logits at coeff 1.0, annealed linearly to 0.0 between 500k and 1M
sampled env steps, then 4M free steps within the 5M budget. No critic
loss term.
"""

from pathlib import Path

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from experiments.rocksample_ppo_2026_09 import shared
from losses.kickstart import KickstartTeacherKLMixin


ENV_CONFIG = {}
D_MODEL = 128
TOTAL_ENV_STEPS = 5_000_000

TEACHER_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "teacher_n5_k7_d64_2m"
)
KICKSTART_COEFF_SCHEDULE = [
    [0, 1.0],
    [500_000, 1.0],
    [1_000_000, 0.0],
]


class KickstartPPOLearner(KickstartTeacherKLMixin, PPOTorchLearner):
    pass


def build_config(context):
    return shared.build_config(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        learner_class=KickstartPPOLearner,
        learner_config_dict={
            "kickstart/coeff": KICKSTART_COEFF_SCHEDULE,
            "kickstart/teacher_checkpoint": str(TEACHER_CHECKPOINT_DIR),
        },
    )


def run(context):
    return shared.run_condition(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        total_env_steps=TOTAL_ENV_STEPS,
        learner_class=KickstartPPOLearner,
        learner_config_dict={
            "kickstart/coeff": KICKSTART_COEFF_SCHEDULE,
            "kickstart/teacher_checkpoint": str(TEACHER_CHECKPOINT_DIR),
        },
    )
