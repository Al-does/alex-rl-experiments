"""Lumped-baseline study rerun from scratch for 4B steps with a per-leaf entropy
anneal: the 0.015 coefficient is held until shortly after each arm's greedy
reward reached its plateau in the seed-42 0->6B runs (parent + cont4b + cont6b),
then decays linearly to 0 over four rollouts (79.7M steps) and stays at 0.

Plateau = median greedy reward over the last 100 checkpoints (4B-6B); onset =
first 2M-step checkpoint within 2% of it that stays there for the next five.
Anneal starts one rollout after onset (rounded to whole rollouts)."""

from experiments.reward_factors_history_jax_lumped_2026_11 import shared as base

COEFF = base.entropy_schedule()[0][1]  # 0.015, the constant-entropy parent coefficient
STUDY = "reward_factors_history_jax_lumped_anneal4b_2026_11"
TOTAL_ENV_STEPS = 4_000_000_000
ANNEAL_WINDOWS = {  # leaf -> (anneal start step, step at which the coefficient reaches 0)
    "reward_both_gru128": (458227712, 537919488),  # plateau 0.588 from 438,304,768
    "reward_both_gru92": (378535936, 458227712),  # plateau 0.588 from 358,612,992
    "reward_both_memoryless": (458227712, 537919488),  # plateau 0.535 from 438,304,768
    "reward_both_transformer": (458227712, 537919488),  # plateau 0.587 from 438,304,768
    "reward_factor_1_gru128": (219152384, 298844160),  # plateau 0.294 from 199,229,440
    "reward_factor_1_gru92": (159383552, 239075328),  # plateau 0.294 from 139,460,608
    "reward_factor_1_memoryless": (99614720, 179306496),  # plateau 0.268 from 79,691,776
    "reward_factor_1_transformer": (338690048, 418381824),  # plateau 0.293 from 318,767,104
}


def entropy_schedule(leaf):
    start, end = ANNEAL_WINDOWS[leaf]
    return ((0, COEFF), (start, COEFF), (end, 0.0))


def run_condition(context, *, condition, architecture):
    leaf = f"{condition}_{architecture}"
    return base.run_condition(
        context, condition=condition, architecture=architecture, total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY, entropy_schedule_points=entropy_schedule(leaf),
    )
