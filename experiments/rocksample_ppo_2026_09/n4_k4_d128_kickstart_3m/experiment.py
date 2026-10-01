"""Width-128 student on RockSample[4,4]: 1M kickstart, then 2M free steps.

Same teacher and schedule as ``n4_k4_d128_kickstart`` (frozen
``n4_k4_d64`` final checkpoint at ``artifacts/teacher_n4_k4_d64/``,
forward KL on policy logits, coeff 1.0 annealed 500k->1M to 0.0), but the
budget is 3M env steps so the student continues for 2M steps after the
kickstart coefficient reaches zero. No critic loss term.
"""

from pathlib import Path

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from experiments.rocksample_ppo_2026_09 import shared
from losses.kickstart import KickstartTeacherKLMixin


ENV_CONFIG = {"n": 4, "k": 4}
D_MODEL = 128
TOTAL_ENV_STEPS = 3_000_000

TEACHER_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "teacher_n4_k4_d64"
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
