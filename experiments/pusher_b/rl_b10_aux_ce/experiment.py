"""rl_b10 with a coefficient-one next-token cross-entropy auxiliary, 116M steps.

Same PPO optimizer recipe as rl_b10 (batch 262,144 / minibatch 8,192 /
num_epochs 6, default lr and entropy anneals), trained from scratch with a
next-token CE auxiliary on the final-block residual stream. Aux logits at
decision time t supervise on the pending token revealed by the delayed
observation at t+1. Matching rl_b10's batch keeps optimizer steps per
iteration identical so the CE comparison is apples-to-apples.

Env layout is sized so one sampling round approximates one batch: 224
runners x 10 vector envs = 2,240 envs x 128 frames = 286,720 steps/round
(vs the 262,144-step target), using all ~224 worker cores on the ~245-CPU
Vast box while leaving ~20 cores for the driver, learner, and Ray overhead.
sample_timeout_s must exceed one full round: timed-out sample() calls are
dropped by the actor manager and orphaned rounds starve later iterations.
"""

from experiments.pusher_b.rl import (
    PPOSettings,
    run_ppo,
)

SETTINGS = PPOSettings(
    total_env_steps=116_000_000,
    train_batch_size=262_144,
    minibatch_size=8_192,
    checkpoint_every_env_steps=20_000_000,
    num_env_runners=224,
    num_envs_per_env_runner=10,
    sample_timeout_s=3600.0,
    next_token_aux=True,
)


def run(context):
    """Fresh 116M-step PPO Pusher-B b=0.10 run with the next-token CE aux."""
    return run_ppo(
        context,
        preset="b10",
        settings=SETTINGS,
    )
