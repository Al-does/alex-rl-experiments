"""Lumped-baseline study rerun from scratch for 4B steps with a per-leaf entropy
anneal: the 0.015 coefficient is held until shortly after each arm's greedy
reward reached its plateau in the seed-42 0->6B runs (parent + cont4b + cont6b),
then decays linearly to 0 over four rollouts (79.7M steps) and stays at 0.

Plateau = median greedy reward over the last 100 checkpoints (4B-6B); onset =
first 2M-step checkpoint within 2% of it that stays there for the next five.
Anneal starts one rollout after onset (rounded to whole rollouts)."""

from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as base

COEFF = base.entropy_schedule()[0][1]  # 0.015, the constant-entropy parent coefficient
STUDY = "belief_quotient_history_jax_lumped_anneal4b_2026_11"
TOTAL_ENV_STEPS = 4_000_000_000
ANNEAL_WINDOWS = {  # leaf -> (anneal start step, step at which the coefficient reaches 0)
    "variant_1_gru128": (99614720, 179306496),  # plateau 0.723 from 79,691,776
    "variant_1_gru92": (59768832, 139460608),  # plateau 0.723 from 39,845,888
    "variant_1_memoryless": (338690048, 418381824),  # plateau 0.723 from 318,767,104
    "variant_1_transformer": (39845888, 119537664),  # plateau 0.723 from 19,922,944
    "variant_2_gru128": (119537664, 199229440),  # plateau 0.32 from 99,614,720
    "variant_2_gru92": (278921216, 358612992),  # plateau 0.32 from 258,998,272
    "variant_2_memoryless": (258998272, 338690048),  # plateau 0.288 from 239,075,328
    "variant_2_transformer": (438304768, 517996544),  # plateau 0.321 from 418,381,824
    "variant_3_gru128": (159383552, 239075328),  # plateau 0.295 from 139,460,608
    "variant_3_gru92": (278921216, 358612992),  # plateau 0.295 from 258,998,272
    "variant_3_memoryless": (278921216, 358612992),  # plateau 0.262 from 258,998,272
    "variant_3_transformer": (398458880, 478150656),  # plateau 0.296 from 378,535,936
}


def entropy_schedule(leaf):
    start, end = ANNEAL_WINDOWS[leaf]
    return ((0, COEFF), (start, COEFF), (end, 0.0))


def run_condition(context, *, variant, architecture):
    leaf = f"variant_{variant}_{architecture}"
    return base.run_condition(
        context, variant=variant, architecture=architecture, total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY, entropy_schedule_points=entropy_schedule(leaf),
    )
