"""Short legacy RLlib PPO run (b10) for the JAX comparison.

Needs ``experiments/pusher_b`` (branch ``devin/1789433175-pusher-b-ntce``).
Same recipe as ``pusher_b/rl_b10`` but stops at ~3M env steps.
"""

TOTAL_ENV_STEPS = 3_000_000


def run(context):
    from experiments.pusher_b import rl

    settings = rl.PPOSettings(total_env_steps=TOTAL_ENV_STEPS)
    return rl.run_ppo(context, preset="b10", settings=settings)
