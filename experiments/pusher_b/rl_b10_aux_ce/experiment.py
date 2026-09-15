"""rl_b10 with a coefficient-one next-token cross-entropy auxiliary, 116M steps.

Same PPO optimizer recipe as the rl_b10 -> rl_b10_continue 116M-step run
(batch 1,048,576 / minibatch 65,536, original lr and entropy anneals), trained
from scratch with a next-token CE auxiliary on the final-block residual
stream. Aux logits at decision time t supervise on the pending token revealed
by the delayed observation at t+1.

Worker count and envs-per-runner are sized for a ~245-CPU Vast box: 224
runners x 37 vector envs = 8,288 envs >= the ~8,256 episodes per 1M-step
batch, i.e. roughly one collection round per iteration while leaving ~20
cores for the driver, learner, and Ray overhead. sample_timeout_s must
exceed one full round: timed-out sample() calls are dropped by the actor
manager and orphaned rounds starve later iterations.
"""

from experiments.pusher_b.rl import (
    PPOSettings,
    run_ppo,
)

SETTINGS = PPOSettings(
    total_env_steps=116_000_000,
    train_batch_size=1_048_576,
    minibatch_size=65_536,
    checkpoint_every_env_steps=20_000_000,
    num_env_runners=224,
    num_envs_per_env_runner=37,
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
