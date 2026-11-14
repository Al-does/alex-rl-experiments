"""Belief steering and single-write pulses on the GRU agents' hidden state.

Sites 0..L-1 are the per-layer hidden states after the cell update (the value
passed up the stack and carried to the next step); site L is the final
LayerNorm embedding, which is not carried. ALL_HIDDEN writes every hidden layer.
The target belief is built from an unsteered shadow network run on the same
inputs, so a write that is itself carried forward is not re-targeted from the
already steered state on the next step.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import affine, target
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_pulse import coordinates, exact_trace, summarize

NONE, ALL_HIDDEN = -1, -2
HIGHEST = jax.lax.Precision.HIGHEST


def empty_maps(spec):
    sites, width = spec.n_layers+1, spec.width
    return dict(decoder=jnp.zeros((sites, width, 2)), bias=jnp.zeros((sites, 2)),
                encoder=jnp.zeros((sites, 2, width)), orthogonal=jnp.zeros((sites, width)))


def decode(h, maps, site):
    y = jnp.matmul(h, maps['decoder'][site], precision=HIGHEST)+maps['bias'][site]
    return jnp.concatenate((y, 1-y.sum(-1, keepdims=True)), -1)


def step(spec, params, carry, obs, maps, site=NONE, mode=0, strength=0., orthogonal=False, reference=None):
    """Match models.step for GRUs; `reference` holds the shadow's site activations."""
    stats = jnp.zeros((obs.shape[0], 3))

    def install(x, index, stats):
        source = x if reference is None else reference[:, index]
        shift = jnp.matmul(target(decode(source, maps, index), mode)[..., :2]-decode(x, maps, index)[..., :2],
                           maps['encoder'][index], precision=HIGHEST)*strength
        norm = jnp.linalg.norm(shift, axis=-1)
        shift = jnp.where(orthogonal, norm[:, None]*maps['orthogonal'][index], shift)
        hnorm = jnp.linalg.norm(x, axis=-1)
        active = (site == index) | ((site == ALL_HIDDEN) & (index < spec.n_layers))
        new_stats = jnp.stack((norm, hnorm, norm/jnp.maximum(hnorm, 1e-12)), -1)
        return jnp.where(active, x+shift, x), jnp.where(active, new_stats, stats)

    x = models._dense(params['input'], obs)
    hidden = []
    for layer, block in enumerate(params['blocks']):
        x = models._gru_cell(block, x, carry[layer])
        x, stats = install(x, layer, stats)
        hidden.append(x)
    embedding = models._layer_norm(params['final_norm'], x)
    embedding, stats = install(embedding, spec.n_layers, stats)
    return models._dense(params['policy'], embedding), jnp.stack(hidden), jnp.stack(hidden+[embedding], 1), stats


@partial(jax.jit, static_argnums=(0, 5, 6, 12))
def rollout(spec, process, params, maps, key, episodes, length, site=NONE, mode=0,
            strength=0., orthogonal=False, random_actions=False, capture=False):
    reset_key, key = jax.random.split(key)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    b = jax.vmap(env.belief_reset, (None, 0))(process, state.token)
    carry = models.init_carry(spec, episodes)

    def body(loop, t):
        state, b, carry, shadow_carry, key = loop
        key, ak, ek = jax.random.split(key, 3)
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        shadow, shadow_carry, shadow_sites, _ = step(spec, params, shadow_carry, obs, maps)
        logits, carry, h, norms = step(spec, params, carry, obs, maps, site, mode, strength, orthogonal, shadow_sites)
        action = jnp.where(random_actions, jax.random.randint(ak, (episodes,), 0, 3), jnp.argmax(logits, -1))
        new_state, reward, _ = jax.vmap(env.step, (None, 0, 0, 0))(process, jax.random.split(ek, episodes), state, action)
        new_b = jax.vmap(env.belief_update, (None, 0, 0, 0))(process, b, action, new_state.token)
        record = dict(reward=reward, action=action, state=state.state, belief=b,
                      changed=action != jnp.argmax(shadow, -1), norms=norms)
        if capture:
            record.update(h=h, obs=obs)
        return (new_state, new_b, carry, shadow_carry, key), record
    return jax.lax.scan(body, (state, b, carry, carry, key), jnp.arange(length))[1]


def _rows(records, name, rows, warmup):
    x = np.asarray(records[name])[warmup:]
    return np.moveaxis(x, 0, 1).reshape((-1,)+x.shape[2:])[:rows]


def fit_maps(spec, records, rows, warmup=8):
    h, b = _rows(records, 'h', rows, warmup), _rows(records, 'belief', rows, warmup)
    maps = {name: [] for name in ('decoder', 'bias', 'encoder', 'orthogonal')}
    rng = np.random.default_rng(42)
    for site in range(spec.n_layers+1):
        d, bias = affine(h[:, site], b[:, :2])
        e, _ = affine(b[:, :2], h[:, site])
        direction = rng.normal(size=spec.width)
        direction -= d @ np.linalg.lstsq(d, direction, rcond=None)[0]
        direction /= np.linalg.norm(direction)
        for name, value in zip(maps, (d, bias, e, direction)):
            maps[name].append(value)
    return {k: jnp.asarray(np.stack(v), dtype=jnp.float32) for k, v in maps.items()}


def probe_scores(spec, records, maps, rows, warmup):
    h, b = _rows(records, 'h', rows, warmup), _rows(records, 'belief', rows, warmup)
    true = np.column_stack((b, b[:, 0]-b[:, 1]))
    variance = ((true-true.mean(0))**2).sum(0)
    scores = []
    for site in range(spec.n_layers+1):
        pred = np.asarray(decode(jnp.asarray(h[:, site]), maps, site))
        pred = np.column_stack((pred, pred[:, 0]-pred[:, 1]))
        r2 = 1-np.divide(((pred-true)**2).sum(0), variance, out=np.full(4, np.nan), where=variance > 0)
        scores.append({k: (float(v) if np.isfinite(v) else None) for k, v in zip(('b1', 'b2', 'rho', 'delta'), r2)})
    return scores


@partial(jax.jit, static_argnums=(0,))
def carries(spec, params, maps, observations, times):
    """Pre-decision hidden state of the intact network at each requested step."""
    carry = models.init_carry(spec, observations.shape[1])
    saved = jnp.zeros((len(times),)+carry.shape)

    def body(loop, item):
        carry, saved = loop
        t, obs = item
        saved = jnp.where((t == times)[:, None, None, None], carry[None], saved)
        return (step(spec, params, carry, obs, maps)[1], saved), None
    return jax.lax.scan(body, (carry, saved), (jnp.arange(len(observations)), observations))[0][1]


@partial(jax.jit, static_argnums=(0,))
def pulse_branch(spec, params, maps, carry, observations, site, mode, orthogonal):
    """One write at lag 0, then identical inputs for the intact and written copies."""
    readout = spec.n_layers-1

    def body(loop, item):
        intact, branch = loop
        k, obs = item
        logits_i, intact, sites_i, _ = step(spec, params, intact, obs, maps)
        logits_b, branch, sites_b, stats = step(spec, params, branch, obs, maps, site, mode,
                                                jnp.where(k == 0, 1., 0.), orthogonal, sites_i)
        out = (decode(sites_b[:, readout], maps, readout)-decode(sites_i[:, readout], maps, readout),
               jnp.abs(jax.nn.softmax(logits_b)-jax.nn.softmax(logits_i)).sum(-1)/2,
               jnp.argmax(logits_b, -1) != jnp.argmax(logits_i, -1), stats[:, 2])
        return (intact, branch), out
    return jax.lax.scan(body, (carry, carry), (jnp.arange(len(observations)), observations))[1]


def tracking(measured, exact):
    """Per-lag agreement of the written copy's decoded change with the exact filter's, across branches."""
    out = []
    for m, e in zip(measured, exact):
        ok = e.std() > 1e-9 and m.std() > 1e-12
        out.append([float(np.corrcoef(m, e)[0, 1]) if ok else None,
                    float((m*e).sum()/(e*e).sum()) if (e*e).sum() > 1e-18 else None])
    return dict(correlation=[x[0] for x in out], slope=[x[1] for x in out])


def run_pulse(spec, params, maps, records, variant, times=(256, 512, 768), horizon=128):
    observations = records['obs']
    saved = carries(spec, params, maps, observations, jnp.asarray(times))
    sites = [(f'layer{i+1}', i) for i in range(spec.n_layers)]+[('all_layers', ALL_HIDDEN)]
    results = {}
    for mode, mode_index, coordinate in (('mirror', 1, 1), ('rho_flip', 2, 0)):
        windows = [np.asarray(observations[t:t+horizon+1]) for t in times]
        beliefs = [np.asarray(records['belief'][t]) for t in times]
        exact = np.concatenate([exact_trace(b, o, variant, mode) for b, o in zip(beliefs, windows)], 1)
        coarse = (np.concatenate([exact_trace(b, o, variant, mode, coarse=True) for b, o in zip(beliefs, windows)], 1)
                  if variant == 2 else None)
        for name, site in sites:
            for control in (False, True):
                parts = [pulse_branch(spec, params, maps, saved[i], observations[t:t+horizon+1], site, mode_index, control)
                         for i, t in enumerate(times)]
                diff, tv, disagree, relative = (np.concatenate([np.asarray(p[j]) for p in parts], 1) for j in range(4))
                measured = coordinates(diff)
                result = dict(measured=summarize(measured), tv=tv.mean(1).tolist(),
                              action_disagreement=disagree.mean(1).tolist(), relative_shift=float(relative[0].mean()))
                if not control:
                    result['full_filter'] = summarize(exact)
                    result['tracking'] = tracking(measured[..., coordinate], exact[..., coordinate])
                    if coarse is not None:
                        result['quotient_filter'] = summarize(coarse)
                results[f"{mode}_{name}{'_orthogonal' if control else ''}"] = result
    return dict(seed=3042, times=list(times), horizon=horizon, readout='top hidden layer decoder', conditions=results)
