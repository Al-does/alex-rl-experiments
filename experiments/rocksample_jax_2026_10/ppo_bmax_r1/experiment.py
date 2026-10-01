"""Round 1 of the large-batch PPO sweep: 8 arms x 4 seeds x 10M env steps.

Question: with the rollout batch pushed far past 65,536 (so each PPO update
averages over 0.26M-2M transitions), which other knobs make the few remaining
updates count? Memory no longer binds after chunking the behaviour-policy
forward (32,768 envs fit in 11 GB), so the batch tiers here are set by the
update budget: 1M (8,192 envs, 10 updates) is the main tier, 2M (5 updates)
the ceiling, 262k (39 updates) a comparator. Knobs: transformer width (64 vs
128), lr, clip, KL target / KL loss, epochs, minibatch, vf_coeff, vf_clip.
Baselines: ``ppo_n5_k7_d64_b65k`` (median tail 23.8 at 5M steps) and the RLlib
PPO plateau of about 25.
"""

from experiments.rocksample_jax_2026_10 import ppo, sweep
from experiments.rocksample_jax_2026_10.sweep import Arm

ENV_STEPS = 10_000_000
NUM_SEEDS = 4

ARMS = (
    Arm(
        "b1m_d64_ref",
        ppo.PPOConfig(num_envs=8192, num_steps=128, num_minibatches=32),
        d_model=64,
        note="naive scale-up of b65k: batch 1M, minibatch 32k, defaults",
    ),
    Arm(
        "b1m_d64_loose",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=64,
        note="512 grad steps/update, lr 1e-3, clip 0.3, KL target 0.05",
    ),
    Arm(
        "b1m_d128_loose",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=128,
        note="as b1m_d64_loose with d_model 128",
    ),
    Arm(
        "b1m_d128_loose_vf1",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05, vf_coeff=1.0,
        ),
        d_model=128,
        note="as b1m_d128_loose with vf_coeff 1.0",
    ),
    Arm(
        "b1m_d128_loose_vfclip100",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05, vf_clip=100.0,
        ),
        d_model=128,
        note="as b1m_d128_loose with squared value error clipped at 100",
    ),
    Arm(
        "b1m_d64_nokl_clip0.5",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=32, num_epochs=8,
            lr=1e-3, clip_param=0.5, use_kl_loss=False,
        ),
        d_model=64,
        note="trust region loosened: clip 0.5, no KL loss, 256 grad steps/update",
    ),
    Arm(
        "b2m_d128_nokl_clip0.5",
        ppo.PPOConfig(
            num_envs=16384, num_steps=128, num_minibatches=64, num_epochs=16,
            lr=1e-3, clip_param=0.5, use_kl_loss=False,
        ),
        d_model=128,
        note="batch 2M: 5 updates, 1,024 grad steps each, clip 0.5, no KL loss",
    ),
    Arm(
        "b262k_d128_loose",
        ppo.PPOConfig(
            num_envs=2048, num_steps=128, num_minibatches=16, num_epochs=4,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=128,
        note="batch 262k comparator: 39 updates, minibatch 16k",
    ),
)

RECIPE = sweep.SweepRecipe(arms=ARMS, env_steps_per_seed=ENV_STEPS, num_seeds=NUM_SEEDS)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
