"""Width-128 student on RockSample[4,4] kickstarted by the width-64 teacher.

The teacher is the final checkpoint of the sibling ``n4_k4_d64`` PPO run
(``n4_k4_d64/results/20260929T205350Z-9e1dfe46``; artifacts on B2 under
``experiments/rocksample_ppo_2026_09/n4_k4_d64/20260929T205350Z-9e1dfe46/``).
Its RLModule checkpoint is staged at ``artifacts/teacher_n4_k4_d64/`` inside
this leaf before launch.

The kickstart loss is forward KL between the frozen teacher's and the
student's per-timestep policy logits at coefficient 1.0, annealed linearly
to 0.0 between 500k and 1M sampled env steps. There is no critic loss term.
"""

from pathlib import Path

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from experiments.rocksample_ppo_2026_09 import shared
from losses.kickstart import KickstartTeacherKLMixin


ENV_CONFIG = {"n": 4, "k": 4}
D_MODEL = 128
TOTAL_ENV_STEPS = 1_000_000

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
