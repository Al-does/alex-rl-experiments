from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10 import ppo, supervised
from experiments.pusher_b_jax_2026_10.process import (
    BOS_TOKEN,
    make_env,
    pusher_b_model,
)

SMALL = tm.ModelSpec(d_model=32, n_layers=2, n_heads=2, d_mlp=64, context_length=16)


def test_kv_cached_step_matches_full_forward():
    params = tm.init_params(SMALL, jax.random.key(0), actor_critic=True)
    tokens = jax.random.randint(jax.random.key(1), (3, 12), 0, 3)
    full = tm.encode(SMALL, params, tokens)
    cache = tm.empty_cache(SMALL, 3)
    steps = []
    for t in range(12):
        embedding, cache = tm.encode_step(SMALL, params, cache, tokens[:, t], jnp.int32(t))
        steps.append(embedding)
    np.testing.assert_allclose(np.stack(steps, 1), full, atol=1e-5)


def test_rollout_tokens_follow_delay_one_semantics():
    env = make_env("b10")
    params = tm.init_params(SMALL, jax.random.key(0), actor_critic=True)
    spec = tm.ModelSpec(**{**SMALL.__dict__, "context_length": 128})
    batch = ppo.collect(env, spec, 4, params, jax.random.key(2))
    tokens, raw = np.asarray(batch.tokens), np.asarray(batch.raw_token)
    assert tokens.shape == (4, 127)
    np.testing.assert_array_equal(tokens[:, 0], BOS_TOKEN)
    np.testing.assert_array_equal(tokens[:, 1:], raw[:, :-1])
    np.testing.assert_array_equal(
        np.asarray(batch.reward), (np.asarray(batch.action) == raw).astype(np.float32)
    )


def test_pusher_b_model_matches_legacy_constants():
    model = pusher_b_model("b10")
    np.testing.assert_allclose(model.transition_matrix.sum(1), 1.0)
    np.testing.assert_allclose(
        model.initial_distribution @ model.transition_matrix, model.initial_distribution
    )


def test_gamma_zero_advantage_is_reward_minus_value():
    config = ppo.PPOConfig.smoke()
    reward = jnp.array([[1.0, 0.0, 1.0]])
    value = jnp.array([[0.2, 0.4, 0.6]])
    advantages, targets = ppo.gae(config, reward, value)
    np.testing.assert_allclose(advantages, reward - value, atol=1e-6)
    np.testing.assert_allclose(targets, reward, atol=1e-6)


def test_schedules_interpolate_like_rllib():
    schedule = ppo.PPOConfig().entropy_schedule
    steps = (0, 6_000_000, 7_000_000, 9_000_000)
    values = [float(ppo.schedule_value(schedule, jnp.int32(s))) for s in steps]
    np.testing.assert_allclose(values, [0.01, 0.01, 0.005, 0.0], atol=1e-7)


@pytest.mark.parametrize("aux", [False, True])
def test_ppo_smoke_runs(aux):
    config = ppo.PPOConfig.smoke(next_token_aux=aux)
    _, curves, summary = ppo.train(
        "b10",
        seed=0,
        config=config,
        spec=tm.ModelSpec(d_model=32, n_layers=1, n_heads=2, d_mlp=64),
        log=lambda _: None,
    )
    assert summary["completed_env_steps"] == 2 * 8 * 127
    assert np.isfinite(curves[-1]["return_mean"])


def test_supervised_smoke_runs():
    result = supervised.train(
        "b10",
        seed=0,
        config=supervised.SupervisedConfig.smoke(),
        spec=tm.ModelSpec(d_model=32, n_layers=1, n_heads=2, d_mlp=64),
        log=lambda _: None,
    )
    assert result["completed_step"] == 4
    assert result["bayesian_floor_nats"] > 0.0


def test_chunked_rollout_and_constant_lr_run():
    config = ppo.PPOConfig.smoke(rollout_chunks=2, lr_schedule=((0, 1e-4),))
    assert float(ppo.schedule_value(config.lr_schedule, jnp.int32(10**7))) == pytest.approx(1e-4)
    _, curves, _ = ppo.train(
        "b10",
        seed=0,
        config=config,
        spec=tm.ModelSpec(d_model=32, n_layers=1, n_heads=2, d_mlp=64),
        log=lambda _: None,
    )
    assert curves[-1]["lr"] == pytest.approx(1e-4)


def test_bfloat16_encode_close_to_float32():
    params = tm.init_params(SMALL, jax.random.key(0), lm_head=True)
    tokens = jax.random.randint(jax.random.key(1), (2, 12), 0, 3)
    full = tm.encode(SMALL, params, tokens)
    half = tm.encode(tm.ModelSpec(**{**SMALL.__dict__, "compute_dtype": "bfloat16"}), params, tokens)
    assert half.dtype == jnp.float32
    np.testing.assert_allclose(half, full, atol=0.1)
