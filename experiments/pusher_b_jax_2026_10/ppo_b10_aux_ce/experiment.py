from experiments.pusher_b_jax_2026_10.ppo import PPOConfig, run_ppo


def run(context):
    return run_ppo(context, preset="b10", config=PPOConfig(next_token_aux=True))
