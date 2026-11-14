"""Experiment-local cached transformer interventions and exact-target rollouts."""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import env, models

MODES = ('point', 'mirror', 'rho_flip', 'rho_point')
LAMBDAS = (0., .25, .4, .45, .5, .55, .6, .75, .9, 1.)
AFFINE_RCOND = 1e-6


def target(b, mode):
    rho = b[..., 2]
    point = jnp.stack((jnp.where(b[..., 0] < b[..., 1], 1-rho, 0),
                       jnp.where(b[..., 0] < b[..., 1], 0, 1-rho), rho), -1)
    mass = b[..., :2].sum(-1, keepdims=True)
    ratio = jnp.where(mass > 0, b[..., :2]/jnp.where(mass > 0, mass, 1), .5)
    def rho_target(r):
        return jnp.concatenate((ratio*(1-r[..., None]), r[..., None]), -1)
    return jax.lax.switch(mode, (lambda: point, lambda: b[..., jnp.array([1, 0, 2])],
                                lambda: rho_target(1-rho),
                                lambda: rho_target((rho < .5).astype(b.dtype))))


def empty_maps(width=64):
    return dict(decoder=jnp.zeros((5, width, 2)), bias=jnp.zeros((5, 2)),
                encoder=jnp.zeros((5, 2, width)), orthogonal=jnp.zeros((5, width)))


def decode(h, maps, site):
    y = jnp.matmul(h, maps['decoder'][site], precision=jax.lax.Precision.HIGHEST) + maps['bias'][site]
    return jnp.concatenate((y, 1-y.sum(-1, keepdims=True)), -1)


def instrumented_step(spec, params, carry, obs, t, maps, site=-1, mode=0,
                      strength=0., orthogonal=False):
    """Match models.step; intervene after the MLP residual or final norm."""
    batch = obs.shape[0]
    cache_len = spec.context_len + 1
    slot = t % cache_len
    valid = jnp.arange(cache_len) <= t
    x = models._dense(params['input'], obs)
    keys, values, sites = [], [], []
    stats = jnp.zeros((batch, 3))

    def install(x, index, stats):
        def apply(x):
            belief = decode(x, maps, index)
            shift = jnp.matmul(target(belief, mode)[..., :2]-belief[..., :2], maps['encoder'][index], precision=jax.lax.Precision.HIGHEST)
            shift = strength*shift
            norm = jnp.linalg.norm(shift, axis=-1)
            shift = jnp.where(orthogonal, norm[:, None]*maps['orthogonal'][index], shift)
            hnorm = jnp.linalg.norm(x, axis=-1)
            return x+shift, jnp.stack((norm, hnorm, norm/jnp.maximum(hnorm, 1e-12)), -1)
        return jax.lax.cond(site == index, apply, lambda x: (x, stats), x)

    for layer, block in enumerate(params['blocks']):
        q, k, v = jnp.split(models._dense(block['qkv'], models._layer_norm(block['ln1'], x)), 3, -1)
        shape = (batch, spec.n_heads, spec.head_dim)
        q = models._rope(q.reshape(shape), t)
        k = models._rope(k.reshape(shape), t)
        kc = carry['k'][layer].at[:, slot].set(k)
        vc = carry['v'][layer].at[:, slot].set(v.reshape(shape))
        scores = jnp.einsum('bhd,bshd->bhs', q, kc)/math.sqrt(spec.head_dim)
        weights = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), -1)
        attended = jnp.einsum('bhs,bshd->bhd', weights, vc).reshape(batch, spec.width)
        x = x + models._dense(block['proj'], attended)
        x = x + models._mlp(block['mlp'], models._layer_norm(block['ln2'], x))
        x, stats = install(x, layer, stats)
        sites.append(x)
        keys.append(kc)
        values.append(vc)
    x = models._layer_norm(params['final_norm'], x)
    x, stats = install(x, 4, stats)
    sites.append(x)
    return models._dense(params['policy'], x), {'k': jnp.stack(keys), 'v': jnp.stack(values)}, jnp.stack(sites, 1), stats


@partial(jax.jit, static_argnums=(0, 5, 6, 12))
def rollout(spec, process, params, maps, key, episodes, length, site=-1, mode=0,
            strength=0., orthogonal=False, random_actions=False, capture=False):
    reset_key, key = jax.random.split(key)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    b = jax.vmap(env.belief_reset, (None, 0))(process, state.token)
    cache = models.init_carry(spec, episodes)

    def body(carry, t):
        state, b, cache, shadow_cache, key = carry
        key, ak, ek = jax.random.split(key, 3)
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        logits, cache, h, norms = instrumented_step(spec, params, cache, obs, t, maps, site, mode, strength, orthogonal)
        shadow, shadow_cache, _, _ = instrumented_step(spec, params, shadow_cache, obs, t, maps)
        action = jnp.where(random_actions, jax.random.randint(ak, (episodes,), 0, 3), jnp.argmax(logits, -1))
        new_state, reward, _ = jax.vmap(env.step, (None, 0, 0, 0))(process, jax.random.split(ek, episodes), state, action)
        new_b = jax.vmap(env.belief_update, (None, 0, 0, 0))(process, b, action, new_state.token)
        record = dict(reward=reward, action=action, state=state.state, belief=b,
                      changed=action != jnp.argmax(shadow, -1), norms=norms)
        if capture:
            record.update(h=h, obs=obs, logits=logits)
        return (new_state, new_b, cache, shadow_cache, key), record
    _, records = jax.lax.scan(body, (state, b, cache, cache, key), jnp.arange(length))
    return records


def affine(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    coef = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=AFFINE_RCOND)[0]
    return coef[:-1], coef[-1]


def fit_maps(records, rows, warmup=8):
    h = np.asarray(records['h'])[warmup:].transpose(1, 0, 2, 3).reshape(-1, 5, 64)[:rows]
    b = np.asarray(records['belief'])[warmup:].transpose(1, 0, 2).reshape(-1, 3)[:rows]
    maps = {name: [] for name in ('decoder', 'bias', 'encoder', 'orthogonal')}
    rng = np.random.default_rng(42)
    for site in range(5):
        d, bias = affine(h[:, site], b[:, :2])
        e, _ = affine(b[:, :2], h[:, site])
        direction = rng.normal(size=64)
        direction -= d @ np.linalg.lstsq(d, direction, rcond=None)[0]
        direction /= np.linalg.norm(direction)
        for name, value in zip(maps, (d, bias, e, direction)):
            maps[name].append(value)
    return {k: jnp.asarray(np.stack(v), dtype=jnp.float32) for k, v in maps.items()}


def probe_scores(records, maps, rows, warmup):
    h = np.asarray(records['h'])[warmup:].transpose(1, 0, 2, 3).reshape(-1, 5, 64)[:rows]
    b = np.asarray(records['belief'])[warmup:].transpose(1, 0, 2).reshape(-1, 3)[:rows]
    true = np.column_stack((b, b[:, 0]-b[:, 1]))
    scores = []
    for site in range(5):
        pred = np.asarray(decode(jnp.asarray(h[:, site]), maps, site))
        pred = np.column_stack((pred, pred[:, 0]-pred[:, 1]))
        variance = ((true-true.mean(0))**2).sum(0)
        r2 = np.divide(((pred-true)**2).sum(0), variance, out=np.full(4, np.nan), where=variance > 0)
        scores.append(dict(zip(('b1', 'b2', 'rho', 'delta'), [None if not np.isfinite(x) else float(x) for x in 1-r2])))
    return scores


def metrics(records):
    r = {k: np.asarray(v) for k, v in records.items()}
    means = r['reward'].mean(0)
    conditional = []
    for s in range(3):
        acts = r['action'][r['state'] == s]
        conditional.append((np.bincount(acts, minlength=3)/len(acts)).tolist() if len(acts) else [None]*3)
    return dict(occupancy=float(means.mean()), occupancy_se=float(means.std(ddof=1)/np.sqrt(len(means))),
                episode_occupancy=means.tolist(), action_fractions=(np.bincount(r['action'].ravel(), minlength=3)/r['action'].size).tolist(),
                action_change=float(r['changed'].mean()), action_given_state=conditional,
                noop_given_reward=conditional[2][0], shift_norm=float(r['norms'][..., 0].mean()),
                activation_norm=float(r['norms'][..., 1].mean()),
                relative_shift=float(r['norms'][..., 0].mean()/max(r['norms'][..., 1].mean(), 1e-12)))


@partial(jax.jit, static_argnums=(0,))
def pulse_branch(spec, params, maps, cache, observations, start, site, mode, strength):
    def body(carry, item):
        k, obs = item
        logits, carry, h, _ = instrumented_step(spec, params, carry, obs, start+k, maps, site, mode, jnp.where(k == 0, strength, 0.))
        return carry, (decode(h[:, 3], maps, 3), jax.nn.softmax(logits), jnp.argmax(logits, -1))
    return jax.lax.scan(body, cache, (jnp.arange(len(observations)), observations))
