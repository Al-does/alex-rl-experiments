import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes as probe


def test_replay_matches_each_original_block_and_final_norm():
    core, _ = probe.frozen_module(probe.STEER_REV, probe.CORE_PATH)
    spec = probe.models.ModelSpec("transformer", 6, 3, 8, 4, context_len=2)
    params = probe.models.init_params(spec, jax.random.PRNGKey(7))
    observations = jax.random.normal(jax.random.PRNGKey(9), (2, 9, 6))
    with jax.default_matmul_precision("highest"):
        actual = np.asarray(probe.replay_sites(spec, params, observations,
                                               core.instrumented_step, core.empty_maps))
        for depth in range(1, 5):
            partial_spec = dataclasses.replace(spec, n_layers=depth)
            partial_params = {**params, "blocks": params["blocks"][:depth]}
            carry = probe.models.init_carry(partial_spec, 2)
            for t in range(9):
                _, residual, embedding, carry = probe.models.step(
                    partial_spec, partial_params, carry, observations[:, t], jnp.asarray(t))
                np.testing.assert_allclose(actual[:, t, depth - 1], residual, atol=1e-6)
                if depth == 4:
                    np.testing.assert_allclose(actual[:, t, 4], embedding, atol=1e-6)


@pytest.mark.parametrize("shape", [(0, 5, 6), (2, 0, 6), (2, 5, 3), (2, 6)])
def test_original_filter_rejects_empty_or_invalid_histories(shape):
    base, _ = probe.frozen_module(probe.ARCH_REV, probe.ARCH_PATH)
    with pytest.raises(ValueError, match="nonempty episode histories"):
        base.merged_filter(np.zeros(shape), {})
