"""As ``…_lumped_b2m_anneal_2026_11`` plus AdamW weight decay 1e-2. Belief-quotient ladder, transformer only, b2m (1,920 envs): 4,000,000,000 steps,
64 minibatches per rollout, softmax temperature annealed 1.5 -> 1.0 over each
leaf's post-plateau window (the windows of ``…_lumped_anneal4b_2026_11``), and
entropy 0.015 held then annealed to 0 over the same window."""

from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as base
from experiments.belief_quotient_history_jax_lumped_anneal4b_2026_11 import shared as anneal

STUDY = "belief_quotient_history_jax_lumped_b2m_anneal_wd_2026_11"
WEIGHT_DECAY = 1e-2  # AdamW decoupled decay, mid-range of the usual 1e-3..1e-1
TOTAL_ENV_STEPS = 4000000000
NUM_ENVS = 1920  # 30 episodes per minibatch (64 minibatches)
ANNEAL_WINDOWS = {k: v for k, v in anneal.ANNEAL_WINDOWS.items() if k.endswith("_transformer")}
COEFF = anneal.COEFF
TEMPERATURE_START, TEMPERATURE_END = 1.5, 1.0


def entropy_schedule(leaf):
    start, end = ANNEAL_WINDOWS[leaf]
    return ((0, COEFF), (start, COEFF), (end, 0.0))


def temperature_schedule(leaf):
    start, end = ANNEAL_WINDOWS[leaf]
    return ((0, TEMPERATURE_START), (start, TEMPERATURE_START), (end, TEMPERATURE_END))


def run_condition(context, *, variant, architecture="transformer"):
    leaf = f"variant_{variant}_{architecture}"
    return base.run_condition(
        context, variant=variant, architecture=architecture, total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY, entropy_schedule_points=entropy_schedule(leaf), temperature=TEMPERATURE_START,
        num_envs=NUM_ENVS, temperature_schedule_points=temperature_schedule(leaf),
        weight_decay=WEIGHT_DECAY,
    )
