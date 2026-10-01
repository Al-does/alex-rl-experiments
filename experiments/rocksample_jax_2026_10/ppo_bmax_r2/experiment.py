"""Round 2 of the large-batch PPO sweep: 6 arms x 4 seeds x 10M env steps.

Round 1 (``ppo_bmax_r1``) showed that at batch 1M every "loose" arm is still
climbing 2-4 return per update at the end of its 10 updates, with entropy still
near 2.0 nats (uniform over 12 actions is 2.48) and the per-update KL (~0.03)
well under the 0.05 target. The budget is therefore bound by how far each of
the 10 updates moves the policy, not by batch noise. Round 2 keeps batch 1M /
d_model 128 (the best, lowest-variance round-1 tier) and pushes per-update
movement: more epochs, higher lr, a weaker entropy bonus, an aggressive
combination, a wider model (d_model 256), and a value clip that actually
binds (vf_clip 10 vs squared value errors of 10-16; the round-1 clip at 100
was a no-op).
"""

from experiments.rocksample_jax_2026_10 import ppo, sweep
from experiments.rocksample_jax_2026_10.sweep import Arm

ENV_STEPS = 10_000_000
NUM_SEEDS = 4

ARMS = (
    Arm(
        "b1m_d128_loose_ep16",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=16,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=128,
        note="as r1 b1m_d128_loose with 16 epochs (1,024 grad steps/update)",
    ),
    Arm(
        "b1m_d128_loose_lr3e-3",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=3e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=128,
        note="as r1 b1m_d128_loose with lr 3e-3",
    ),
    Arm(
        "b1m_d128_loose_ent0.01",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05, entropy_coeff=0.01,
        ),
        d_model=128,
        note="as r1 b1m_d128_loose with entropy_coeff 0.01",
    ),
    Arm(
        "b1m_d128_push",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=16,
            lr=1e-3, clip_param=0.5, kl_target=0.1, entropy_coeff=0.01,
        ),
        d_model=128,
        note="combined: 16 epochs, clip 0.5, KL target 0.1, entropy_coeff 0.01",
    ),
    Arm(
        "b1m_d256_loose",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05,
        ),
        d_model=256,
        note="as r1 b1m_d128_loose with d_model 256",
    ),
    Arm(
        "b1m_d128_loose_vfclip10",
        ppo.PPOConfig(
            num_envs=8192, num_steps=128, num_minibatches=64, num_epochs=8,
            lr=1e-3, clip_param=0.3, kl_target=0.05, vf_clip=10.0,
        ),
        d_model=128,
        note="as r1 b1m_d128_loose with squared value error clipped at 10",
    ),
)

RECIPE = sweep.SweepRecipe(arms=ARMS, env_steps_per_seed=ENV_STEPS, num_seeds=NUM_SEEDS)


def recipe(context) -> sweep.SweepRecipe:
    return sweep.smoke_recipe() if context.smoke else RECIPE


def run(context):
    return sweep.run(context, recipe(context))
