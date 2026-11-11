"""Belief-quotient ladder, transformer only, b20m (192 envs), entropy 0: 1,000,000,000 steps,
64 minibatches per rollout, softmax temperature annealed 1.5 -> 1.0 over each
leaf's post-plateau window (the windows of ``…_lumped_anneal4b_2026_11``), and
entropy 0.0 throughout."""

from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as base
from experiments.belief_quotient_history_jax_lumped_anneal4b_2026_11 import shared as anneal

STUDY = "belief_quotient_history_jax_lumped_b20m_temp_anneal_2026_11"
TOTAL_ENV_STEPS = 1000000000
NUM_ENVS = 192  # 3 episodes per minibatch (64 minibatches)
ANNEAL_WINDOWS = {k: v for k, v in anneal.ANNEAL_WINDOWS.items() if k.endswith("_transformer")}
COEFF = anneal.COEFF
TEMPERATURE_START, TEMPERATURE_END = 1.5, 1.0


def entropy_schedule(leaf):
    start, end = ANNEAL_WINDOWS[leaf]
    return ((0, 0.0),)


def temperature_schedule(leaf):
    start, end = ANNEAL_WINDOWS[leaf]
    return ((0, TEMPERATURE_START), (start, TEMPERATURE_START), (end, TEMPERATURE_END))


def run_condition(context, *, variant, architecture="transformer"):
    leaf = f"variant_{variant}_{architecture}"
    return base.run_condition(
        context, variant=variant, architecture=architecture, total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY, entropy_schedule_points=entropy_schedule(leaf), temperature=TEMPERATURE_START,
        num_envs=NUM_ENVS, temperature_schedule_points=temperature_schedule(leaf),
    )
