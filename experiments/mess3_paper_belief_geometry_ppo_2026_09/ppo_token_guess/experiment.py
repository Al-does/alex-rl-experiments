"""Gamma-zero PPO on the paper's model and data, rewarded for correct guesses."""

from harness.context import RunContext

from ..ppo import PPOConfig
from ..shared import run_ppo_token_guess


# 2M agent steps: 50 rollouts of 4,000 fresh sequences (40,000 scored
# positions), six epochs of 500-sequence minibatches (2,400 Adam updates).
FULL_PPO_CONFIG = PPOConfig()
SMOKE_PPO_CONFIG = PPOConfig.smoke()


def run(context: RunContext):
    return run_ppo_token_guess(
        context,
        full_config=FULL_PPO_CONFIG,
        smoke_config=SMOKE_PPO_CONFIG,
    )
