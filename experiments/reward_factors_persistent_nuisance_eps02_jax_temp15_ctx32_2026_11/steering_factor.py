"""Single-factor belief steering in the two-factor epsilon-0.02 transformers.

Joint state is ``3*s1+s2``. A write moves one factor's decoded belief to a
counterfactual target and leaves the other factor's decoded belief where it is.
``marginal`` maps decode the first two coordinates of each factor; ``joint``
maps decode the first eight joint probabilities and target the outer product of
the two factor beliefs.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import (
    AFFINE_RCOND, affine, instrumented_step,
)
from experiments.reward_factors_history_jax_2026_11.process import ACTION_PAIRS

MODES = ('point', 'mirror', 'rho_flip', 'corner')
KINDS = ('marginal', 'joint')
SITES = 5
PAIRS = np.asarray(ACTION_PAIRS)


def targets(b, xp=jnp):
    """All four single-factor targets, stacked on axis 0 in ``MODES`` order."""
    rho = b[..., 2]
    low = b[..., 0] < b[..., 1]
    zero = xp.zeros_like(rho)
    point = xp.stack((xp.where(low, 1-rho, zero), xp.where(low, zero, 1-rho), rho), -1)
    mass = b[..., :2].sum(-1, keepdims=True)
    ratio = xp.where(mass > 0, b[..., :2]/xp.where(mass > 0, mass, 1), .5)
    flip = xp.concatenate((ratio*rho[..., None], 1-rho[..., None]), -1)
    corner = xp.stack((low*xp.ones_like(rho), (~low)*xp.ones_like(rho), zero), -1)
    return xp.stack((point, b[..., xp.array([1, 0, 2])], flip, corner))


def split(mode):
    return mode // len(MODES), mode % len(MODES)


def full(y):
    return jnp.concatenate((y, 1-y.sum(-1, keepdims=True)), -1)


def decoded(x, maps, index):
    return jnp.matmul(x, maps['decoder'][index], precision=jax.lax.Precision.HIGHEST)+maps['bias'][index]


def marginal_displacement(x, maps, index, mode):
    factor, kind = split(mode)
    y = decoded(x, maps, index)
    beliefs = jnp.stack((full(y[..., :2]), full(y[..., 2:])))
    change = (targets(beliefs[factor])[kind]-beliefs[factor])[..., :2]
    zero = jnp.zeros_like(change)
    return jnp.concatenate((jnp.where(factor == 0, change, zero), jnp.where(factor == 1, change, zero)), -1)


def joint_displacement(x, maps, index, mode):
    factor, kind = split(mode)
    joint = full(decoded(x, maps, index)).reshape(-1, 3, 3)
    beliefs = jnp.stack((joint.sum(2), joint.sum(1)))
    moved = targets(beliefs[factor])[kind]
    first = jnp.where(factor == 0, moved, beliefs[0])
    second = jnp.where(factor == 1, moved, beliefs[1])
    return (first[:, :, None]*second[:, None, :]-joint).reshape(-1, 9)[:, :8]


DISPLACEMENTS = dict(marginal=marginal_displacement, joint=joint_displacement)


@partial(jax.jit, static_argnums=(0, 5, 6, 12, 13))
def rollout(spec, process, params, maps, key, episodes, length, site=-1, mode=0,
            strength=0., orthogonal=False, random_actions=False, capture=False, kind='marginal'):
    displacement = DISPLACEMENTS[kind]
    reset_key, key = jax.random.split(key)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    b = jax.vmap(env.belief_reset, (None, 0))(process, state.token)
    cache = models.init_carry(spec, episodes)
    pairs = jnp.asarray(PAIRS)

    def body(carry, t):
        state, b, cache, shadow_cache, key = carry
        key, ak, ek = jax.random.split(key, 3)
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        logits, cache, h, norms = instrumented_step(spec, params, cache, obs, t, maps, site, mode, strength,
                                                    orthogonal, displacement)
        shadow, shadow_cache, _, _ = instrumented_step(spec, params, shadow_cache, obs, t, maps,
                                                       displacement=displacement)
        action = jnp.where(random_actions, jax.random.randint(ak, (episodes,), 0, len(ACTION_PAIRS)), jnp.argmax(logits, -1))
        new_state, _, _ = jax.vmap(env.step, (None, 0, 0, 0))(process, jax.random.split(ek, episodes), state, action)
        new_b = jax.vmap(env.belief_update, (None, 0, 0, 0))(process, b, action, new_state.token)
        record = dict(action=action, state=state.state, belief=b,
                      changed=pairs[action] != pairs[jnp.argmax(shadow, -1)], norms=norms)
        if capture:
            record.update(h=h)
        return (new_state, new_b, cache, shadow_cache, key), record
    _, records = jax.lax.scan(body, (state, b, cache, cache, key), jnp.arange(length))
    return records


def marginals(joint):
    joint = np.asarray(joint, float).reshape(joint.shape[:-1]+(3, 3))
    return joint.sum(-1), joint.sum(-2)


def coordinates(joint, kind):
    if kind == 'joint':
        return joint[:, :8]
    first, second = marginals(joint)
    return np.column_stack((first[:, :2], second[:, :2]))


def flatten(records, rows, warmup):
    h = np.asarray(records['h'])[warmup:]
    h = h.transpose(1, 0, 2, 3).reshape(-1, SITES, h.shape[-1])[:rows]
    b = np.asarray(records['belief'])[warmup:].transpose(1, 0, 2).reshape(-1, 9)[:rows]
    return h, b


def empty_maps(kind, width=64):
    k = 4 if kind == 'marginal' else 8
    return dict(decoder=jnp.zeros((SITES, width, k)), bias=jnp.zeros((SITES, k)),
                encoder=jnp.zeros((SITES, k, width)), orthogonal=jnp.zeros((SITES, width)))


def fit_maps(records, rows, kind, warmup=8):
    h, b = flatten(records, rows, warmup)
    y = coordinates(b, kind)
    maps = {name: [] for name in ('decoder', 'bias', 'encoder', 'orthogonal')}
    rng = np.random.default_rng(42)
    for site in range(SITES):
        d, bias = affine(h[:, site], y)
        e, _ = affine(y, h[:, site])
        direction = rng.normal(size=h.shape[-1])
        direction -= d @ np.linalg.lstsq(d, direction, rcond=None)[0]
        direction /= np.linalg.norm(direction)
        for name, value in zip(maps, (d, bias, e, direction)):
            maps[name].append(value)
    return {k: jnp.asarray(np.stack(v), dtype=jnp.float32) for k, v in maps.items()}


def probe_scores(records, maps, rows, kind, warmup=8):
    """Held-out R² of each factor's rho and delta read through this map."""
    h, b = flatten(records, rows, warmup)
    first, second = marginals(b)
    true = np.column_stack((first[:, 2], first[:, 0]-first[:, 1], second[:, 2], second[:, 0]-second[:, 1]))
    scores = []
    for site in range(SITES):
        y = np.asarray(decoded(jnp.asarray(h[:, site]), maps, site), float)
        if kind == 'joint':
            y = coordinates(np.column_stack((y, 1-y.sum(-1))), 'marginal')
        pred = np.column_stack((1-y[:, 0]-y[:, 1], y[:, 0]-y[:, 1], 1-y[:, 2]-y[:, 3], y[:, 2]-y[:, 3]))
        variance = ((true-true.mean(0))**2).sum(0)
        r2 = 1-((pred-true)**2).sum(0)/variance
        scores.append(dict(zip(('rho_1', 'delta_1', 'rho_2', 'delta_2'), map(float, r2))))
    return scores


def metrics(records, policy=None, mode=0, strength=0., steered=False):
    r = {k: np.asarray(v) for k, v in records.items()}
    state = np.stack(np.divmod(r['state'], 3), -1)
    occupied = (state == 2).mean(0)
    episodes = len(occupied)
    result = dict(occupancy=occupied.mean(0).tolist(),
                  occupancy_se=(occupied.std(0, ddof=1)/np.sqrt(episodes)).tolist(),
                  action_change=r['changed'].mean((0, 1)).tolist(),
                  shift_norm=float(r['norms'][..., 0].mean()), activation_norm=float(r['norms'][..., 1].mean()),
                  relative_shift=float(r['norms'][..., 0].mean()/max(r['norms'][..., 1].mean(), 1e-12)))
    if policy is not None:
        factor, kind = split(mode)
        installed = list(marginals(r['belief']))
        if steered:
            b = installed[factor]
            installed[factor] = (1-strength)*b+strength*targets(b, np)[kind]
        executed = PAIRS[r['action']]
        result['reference_action_agreement'] = [float((policy(installed[f]) == executed[..., f]).mean()) for f in (0, 1)]
    return result
