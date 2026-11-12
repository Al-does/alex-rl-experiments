"""Small-batch (b2m) temp15 recipes: batch geometry and inherited settings."""

import pytest

from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as lumped
from experiments.belief_quotient_history_jax_lumped_temp15_b2m_2026_11 import shared as belief
from experiments.reward_factors_history_jax_lumped_temp15_b2m_2026_11 import shared as reward


@pytest.mark.parametrize("pkg", [belief, reward])
def test_batch_is_tenth_of_parent_with_same_minibatch_count(pkg):
    config = pkg.base.train_config(False, pkg.TOTAL_ENV_STEPS, pkg.ENTROPY_SCHEDULE, pkg.TEMPERATURE, pkg.NUM_ENVS)
    parent = pkg.base.train_config(False)
    assert config.num_minibatches == parent.num_minibatches == 64
    assert config.num_envs == 1_920 and config.num_envs % config.num_minibatches == 0
    assert config.batch_steps == 1_966_080
    assert 0.09 < config.batch_steps / parent.batch_steps < 0.11
    assert 0.09 < (config.num_envs // 64) / (parent.num_envs // 64) < 0.11
    assert config.total_env_steps == 500_000_000 and config.iterations == 255
    assert config.temperature == 1.5 and config.entropy_schedule == ((0, 0.0),)
    assert config.episode_length == parent.episode_length and config.lr == parent.lr


def test_default_batch_unchanged():
    assert lumped.train_config(False).num_envs == 19_456
