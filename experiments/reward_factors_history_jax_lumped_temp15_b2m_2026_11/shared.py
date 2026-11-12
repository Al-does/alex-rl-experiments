"""Small-batch variant of the temp15 recipe: entropy 0, sampling temperature 1.5,
500M env steps, and a ~10x smaller synchronized batch -- 1,920 envs x 1,024
steps = 1,966,080 steps per rollout (parent: 19,456 envs, 19,922,944 steps),
still split into 64 minibatches (30 episodes = 30,720 steps each, parent 304
episodes), so there are ~10x more Adam steps per env step (255 rollouts =
16,320 updates at 500M)."""

from experiments.reward_factors_history_jax_lumped_2026_11 import shared as base
from experiments.reward_factors_history_jax_lumped_temp15_2026_11 import shared as parent

STUDY = "reward_factors_history_jax_lumped_temp15_b2m_2026_11"
TOTAL_ENV_STEPS = 500_000_000
NUM_ENVS = 1_920  # 30 x NUM_MINIBATCHES (64); ~1/10 of the parent batch
ENTROPY_SCHEDULE = parent.ENTROPY_SCHEDULE
TEMPERATURE = parent.TEMPERATURE


def run_condition(context, *, condition, architecture):
    return base.run_condition(
        context, condition=condition, architecture=architecture, total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY, entropy_schedule_points=ENTROPY_SCHEDULE, temperature=TEMPERATURE,
        num_envs=NUM_ENVS,
    )
