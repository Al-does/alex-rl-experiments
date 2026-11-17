"""Unmodified function extracts: deep-beliefs@177ba8d9081933b83a3834b7a1f92ac31bb3f02b experiments/persistent_nuisance_v2/reprobe.py."""

import numpy as np
from functools import partial, lru_cache


def lumped_model(process):
    emission = np.asarray(process["emission"], dtype=np.float64)
    transitions = np.asarray(process["transitions"], dtype=np.float64)
    e = np.stack([emission[:, :2].sum(1), emission[:, 2]], axis=-1)
    t = np.stack([transitions[:, :, :2].sum(2), transitions[:, :, 2]], axis=-1)
    if not np.allclose(t[:, 0], t[:, 1], atol=1e-8, rtol=0):
        raise ValueError("not strongly lumpable over states 1 and 2")
    if not np.allclose(e[0], e[1], atol=1e-8, rtol=0):
        raise ValueError("coarsened emissions differ")
    initial = np.asarray(process["initial"], dtype=np.float64)
    return np.array([initial[:2].sum(), initial[2]]), e[[0, 2]], t[:, [0, 2]]


def coarse_filter(observations, process):
    """Complete episode prefixes; action is the preceding executed action."""
    initial, emission, transitions = lumped_model(process)
    tokens = observations[..., :3].argmax(-1)
    actions = observations[..., 3:].argmax(-1)
    belief = np.broadcast_to(initial, (len(observations), 2)).copy()
    out = np.empty(observations.shape[:2], dtype=np.float64)
    for t in range(observations.shape[1]):
        if t:
            belief = np.einsum("bi,bij->bj", belief, transitions[actions[:, t]])
        belief *= emission[:, (tokens[:, t] == 2).astype(int)].T
        belief /= belief.sum(1, keepdims=True)
        out[:, t] = belief[:, 1]
    return out


@lru_cache(maxsize=1)
def runtime():
    import jax
    import jax.numpy as jnp
    from experiments.belief_quotient_history_jax_2026_11 import models
    from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout

    @partial(jax.jit, static_argnums=(0, 3))
    def replay(spec, params, observations, site):
        carry = models.init_carry(spec, len(observations))
        def step(carry, inputs):
            t, obs = inputs
            _, residual, embedding, carry = models.step(spec, params, carry, obs, t)
            return carry, embedding if site == "embedding" else residual
        _, features = jax.lax.scan(step, carry, (jnp.arange(observations.shape[1]), jnp.swapaxes(observations, 0, 1)))
        return jnp.swapaxes(features, 0, 1)
    return jax, _probe_rollout, replay


def flatten(value, warmup, rows):
    v = np.asarray(value)[:, warmup:]
    return v.reshape(-1, *v.shape[2:])[:rows]


def fit_score(x, z, y, target):
    from analysis.probes.linear import fit_affine_probe, global_mse_metrics, r2_score
    weight, bias = fit_affine_probe(x, y, ridge=1e-6)
    predicted = np.asarray(z, dtype=np.float64) @ weight + bias
    metrics = global_mse_metrics(predicted, target)
    metrics["r_squared"] = r2_score(predicted, target)
    metrics["error"] = 1 - metrics["r_squared"]
    return metrics, predicted
