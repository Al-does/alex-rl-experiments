"""Round 5: single-seed, seed-matched one-knob variations of the ep16 recipe.

Control is round 3's ``b1m_d128_loose_ep16`` seed 0 (43.5 at 30M); every arm
here runs seed 0 with the same key (``num_keys=4``) and changes exactly one
thing: width (64/192; 256 was 39.2 in round 4), depth (2/4 layers), context
(16/64), clip 0.5 or KL target 0.1 alone (round 2's ``push`` moved both plus
entropy), and more gradient steps per update (32 epochs, or 128 minibatches
of 8k). Winners get a few-seed confirmation against the control afterwards.
"""

from dataclasses import replace

from experiments.rocksample_jax_2026_10 import ppo, sweep
from experiments.rocksample_jax_2026_10.sweep import Arm

ENV_STEPS = 30_000_000
NUM_SEEDS = 1

BASE = ppo.PPOConfig(
    num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=16,
    lr=1e-3, clip_param=0.3, kl_target=0.05,
)

ARMS = (
    Arm("d192", BASE, d_model=192, note="width 192 (control d128 43.5, d256 39.2)"),
    Arm("d64", BASE, d_model=64, note="width 64 with the ep16 recipe"),
    Arm("d128_l2", BASE, d_model=128, n_layers=2, note="2 layers (lookback 64)"),
    Arm("d128_l4", BASE, d_model=128, n_layers=4, note="4 layers (lookback 128)"),
    Arm("d128_ctx16", BASE, d_model=128, context_len=16, note="context 16 (lookback 48)"),
    Arm("d128_ctx64", BASE, d_model=128, context_len=64, note="context 64 (lookback 192)"),
    Arm("d128_clip0.5", replace(BASE, clip_param=0.5), d_model=128, note="clip 0.5 alone"),
    Arm("d128_kl0.1", replace(BASE, kl_target=0.1), d_model=128, note="KL target 0.1 alone"),
    Arm("d128_ep32", replace(BASE, num_epochs=32), d_model=128, note="32 epochs (2,048 grad steps/update)"),
    Arm(
        "d128_mb8k",
        replace(BASE, num_minibatches=128),
        d_model=128,
        note="128 minibatches of 8k (2,048 grad steps/update, same epochs)",
    ),
)

RECIPE = sweep.SweepRecipe(
    arms=ARMS,
    env_steps_per_seed=ENV_STEPS,
    num_seeds=NUM_SEEDS,
    num_keys=4,
)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
