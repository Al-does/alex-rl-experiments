import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
if not Path(os.environ.get("PULSE_REFERENCE", HERE.parent / "artifacts" / "arch_pulse_reference")).exists():
    pytest.skip("Prepare the pinned reference checkout documented in README.md", allow_module_level=True)

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse import recurrent as r
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse import write_modes as w
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse.reference import core

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models


def fixture(layers):
    spec = models.ModelSpec("gru", 6, 3, 8, layers)
    params = models.init_params(spec, jax.random.PRNGKey(4))
    initial = jax.random.normal(jax.random.PRNGKey(7), (layers, 2, 8)) * .1
    obs = jax.random.normal(jax.random.PRNGKey(3), (2, 6))
    maps = r.empty_maps(spec)
    for i in range(layers):
        maps["decoder"] = maps["decoder"].at[i, :2].set(jnp.eye(2) * (.5+.1*i))
        maps["encoder"] = maps["encoder"].at[i, :, :2].set(jnp.eye(2) * (.8+.05*i))
        maps["bias"] = maps["bias"].at[i].set(jnp.array([.25, .35]))
    return spec, params, initial, obs, maps


@pytest.mark.parametrize("layers", [2, 4])
@pytest.mark.parametrize("mode", [1, 2])
def test_every_layer_is_written_in_order_and_carried(layers, mode):
    spec, params, initial, obs, maps = fixture(layers)
    _, intact, reference = r.step(spec, params, initial, obs, 0, maps)
    actual, sites = w.all_layer_step(spec, params, initial, obs, maps, reference, mode, 1.)
    x = models._dense(params["input"], obs)
    expected, raw_states = [], []
    for layer, block in enumerate(params["blocks"]):
        raw = models._gru_cell(block, x, initial[layer])
        raw_states.append(raw)
        desired = core.target(core.decode(reference[:, layer], maps, layer), mode)
        displacement = desired[..., :2]-core.decode(raw, maps, layer)[..., :2]
        x = raw + displacement @ maps["encoder"][layer]
        expected.append(x)
        assert not np.allclose(x, raw)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_array_equal(sites, np.swapaxes(actual, 0, 1))
    for layer in range(1, layers):
        wrong_input = models._gru_cell(params["blocks"][layer], intact[layer-1], initial[layer])
        assert not np.allclose(raw_states[layer], wrong_input)
    _, next_intact, next_reference = r.step(spec, params, intact, obs, 1, maps)
    next_actual, _ = w.all_layer_step(spec, params, actual, obs, maps, next_reference, mode, 0.)
    expected_next = models.step(spec, params, actual, obs, 1)[-1]
    np.testing.assert_allclose(next_actual, expected_next, atol=1e-6)
    for layer in range(layers):
        wrong_carry = actual.at[layer].set(intact[layer])
        wrong_next = models.step(spec, params, wrong_carry, obs, 1)[-1]
        assert not np.allclose(next_actual[layer], wrong_next[layer])
    assert not np.allclose(next_actual, next_intact)


@pytest.mark.parametrize("layers", [2, 4])
def test_zero_strength_matches_trainer(layers):
    spec, params, initial, obs, maps = fixture(layers)
    _, intact, reference = r.step(spec, params, initial, obs, 0, maps)
    actual, _ = w.all_layer_step(spec, params, initial, obs, maps, reference, 2, 0.)
    np.testing.assert_allclose(actual, intact, atol=1e-6)


def test_targets_are_from_shadow_not_already_steered_layer_output():
    spec, params, initial, obs, maps = fixture(4)
    _, _, reference = r.step(spec, params, initial, obs, 0, maps)
    actual, _ = w.all_layer_step(spec, params, initial, obs, maps, reference, 2, 1.)
    x = models._dense(params["input"], obs)
    self_targeted = []
    for layer, block in enumerate(params["blocks"]):
        raw = models._gru_cell(block, x, initial[layer])
        b = core.decode(raw, maps, layer)
        x = raw + (core.target(b, 2)[..., :2]-b[..., :2]) @ maps["encoder"][layer]
        self_targeted.append(x)
    np.testing.assert_allclose(actual[0], self_targeted[0], atol=1e-6)
    assert not np.allclose(actual[-1], self_targeted[-1])


@pytest.mark.parametrize("layers", [2, 4])
@pytest.mark.parametrize("mode", [1, 2])
def test_branch_writes_once_and_replays_same_inputs(layers, mode):
    spec, params, initial, obs, maps = fixture(layers)
    observations = jnp.stack((obs, obs[:, ::-1], obs, obs[:, ::-1]))
    actual = w.all_layer_branch(spec, params, maps, initial, observations, mode)
    intact, written, expected = initial, initial, []
    for lag, observation in enumerate(observations):
        _, intact, reference = r.step(spec, params, intact, observation, lag, maps)
        written, sites = w.all_layer_step(spec, params, written, observation, maps, reference, mode, float(lag == 0))
        expected.append(core.decode(sites[:, -1], maps, layers-1)-core.decode(reference[:, -1], maps, layers-1))
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("times,horizon", [((), 1), ((-1,), 1), ((1,), -1), ((2,), 2)])
def test_invalid_windows(times, horizon):
    spec, params, _, obs, maps = fixture(2)
    with pytest.raises(ValueError, match="pulse times"):
        w.run_all_layers(spec, params, maps, {"obs": jnp.broadcast_to(obs, (4, 2, 6))}, times, horizon)
