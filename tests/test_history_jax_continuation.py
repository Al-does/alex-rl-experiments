"""Continuation of JAX history runs from parameter checkpoints."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import continuation, models, training
from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig, train
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as lumped


def _setup():
    process = lumped.ladder_process(2)
    spec = lumped.model_spec("gru92", process)
    config = TrainConfig(total_env_steps=2 * 2 * 64, num_envs=2, episode_length=64,
                         num_minibatches=2, entropy_schedule=lumped.entropy_schedule())
    return process, spec, config


def test_npz_round_trip(tmp_path):
    process, spec, _ = _setup()
    params = models.init_params(spec, jax.random.PRNGKey(3))
    training._save(tmp_path / "p.npz", jax.device_get(params))
    loaded = continuation.load_params(tmp_path / "p.npz", spec)
    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(params)
    for a, b in zip(jax.tree_util.tree_leaves(loaded), jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_continuation_starts_from_given_params_with_absolute_steps():
    process, spec, config = _setup()
    seen = {}
    first, _, summary = train(spec, process, config, seed=7,
                              checkpoint_steps=training.checkpoint_steps(config),
                              on_checkpoint=lambda s, p: seen.setdefault(("a", s), p))
    assert summary["env_steps"] == 256 and summary["step_offset"] == 0
    offset = summary["env_steps"]
    cont_seen = {}
    _, curves, cont = train(spec, process, config, seed=7, initial_params=first,
                            step_offset=offset,
                            checkpoint_steps=training.checkpoint_steps(config, offset),
                            on_checkpoint=lambda s, p: cont_seen.setdefault(s, p))
    assert cont["step_offset"] == offset and cont["env_steps"] == 2 * offset
    assert min(cont_seen) == offset and max(cont_seen) == 2 * offset
    assert [c["env_steps"] for c in curves] == [offset + 128, offset + 256]
    for a, b in zip(jax.tree_util.tree_leaves(cont_seen[offset]), jax.tree_util.tree_leaves(first)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_checkpoint_steps_offset():
    config = TrainConfig(total_env_steps=4 * 1024 * 2, num_envs=2, episode_length=1024)
    base = training.checkpoint_steps(config)
    assert training.checkpoint_steps(config, 10_000 * 2048) == [10_000 * 2048 + s for s in base]
