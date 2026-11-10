"""Temperature schedule in the JAX trainer and the transformer-only anneal packages."""

import pytest

from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig, entropy_coefficient, train
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as lumped
from experiments.belief_quotient_history_jax_lumped_b2m_anneal_2026_11 import shared as b2m
from experiments.belief_quotient_history_jax_lumped_b20m_anneal_2026_11 import shared as b20m
from experiments.belief_quotient_history_jax_lumped_b20m_temp_anneal_2026_11 import shared as b20m_t


def test_scheduled_temperature_is_recorded_per_iteration():
    process = lumped.ladder_process(2)
    spec = lumped.model_spec("memoryless", process)
    config = TrainConfig(total_env_steps=4 * 64, num_envs=4, episode_length=16, num_minibatches=2,
                         temperature=1.5, temperature_schedule=((0, 1.5), (64, 1.5), (192, 1.0)))
    _, curves, _ = train(spec, process, config, seed=3)
    assert [round(c["temperature"], 4) for c in curves] == [1.5, 1.5, 1.25, 1.0]


@pytest.mark.parametrize("pkg,nenv,total,entropy_on", [(b2m, 1920, 4_000_000_000, True), (b20m, 192, 1_000_000_000, True), (b20m_t, 192, 1_000_000_000, False)])
def test_packages(pkg, nenv, total, entropy_on):
    assert set(pkg.ANNEAL_WINDOWS) == {"variant_1_transformer", "variant_2_transformer", "variant_3_transformer"}
    for leaf, (start, end) in pkg.ANNEAL_WINDOWS.items():
        cfg = pkg.base.train_config(False, pkg.TOTAL_ENV_STEPS, pkg.entropy_schedule(leaf), pkg.TEMPERATURE_START, pkg.NUM_ENVS, pkg.temperature_schedule(leaf))
        assert cfg.num_envs == nenv and cfg.num_minibatches == 64 and cfg.total_env_steps == total
        assert entropy_coefficient(cfg.temperature_schedule, start) == 1.5
        assert entropy_coefficient(cfg.temperature_schedule, (start + end) // 2) == pytest.approx(1.25)
        assert entropy_coefficient(cfg.temperature_schedule, total) == 1.0
        assert entropy_coefficient(cfg.entropy_schedule, start) == (0.015 if entropy_on else 0.0)
        assert entropy_coefficient(cfg.entropy_schedule, total) == 0.0
