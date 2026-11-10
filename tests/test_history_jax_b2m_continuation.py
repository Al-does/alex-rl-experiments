"""Small-batch temp15 continuation (b2m_cont2b): geometry of the +1.5B stretch."""

import pytest

from experiments.belief_quotient_history_jax_2026_11.training import checkpoint_steps
from experiments.belief_quotient_history_jax_lumped_temp15_b2m_cont2b_2026_11 import shared as belief
from experiments.reward_factors_history_jax_lumped_temp15_b2m_cont2b_2026_11 import shared as reward

PARENT_FINAL = 501_350_400


@pytest.mark.parametrize("pkg", [belief, reward])
def test_continuation_config_matches_parent_batch_and_adds_1p5b(pkg):
    cfg = pkg.base.train_config(False, pkg.ADDITIONAL_ENV_STEPS, pkg.parent.ENTROPY_SCHEDULE,
                                pkg.parent.TEMPERATURE, pkg.parent.NUM_ENVS)
    assert cfg.num_envs == 1_920 and cfg.num_minibatches == 64 and cfg.temperature == 1.5
    assert cfg.entropy_schedule == ((0, 0.0),)
    assert PARENT_FINAL % cfg.batch_steps == 0  # parent ended on a whole rollout
    assert cfg.iterations == 763
    assert PARENT_FINAL + cfg.iterations * cfg.batch_steps == 2_001_469_440
    steps = checkpoint_steps(cfg, PARENT_FINAL)
    assert min(steps) > PARENT_FINAL and max(steps) < 2_001_469_440
