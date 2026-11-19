"""Action-fixed counterfactual history replay and target-conditioned writes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from experiments.algorithm_generality_probes_eps02_2026_11.steer_moving_base import forward
from experiments.belief_quotient_history_jax_2026_11 import env, models

SITE = 3
FIT_SEED, TEST_SEED, TUNE_SEED, CONFIRM_SEED = 42, 43, 1042, 2042
RCOND = 1e-6
REGIME_LENGTH = 64


def edit_histories(obs, emission, factors, seed):
    """Generate alternative evidence; preserve reset and action encoding."""
    rng = np.random.default_rng(seed)
    token_count = 3 ** factors
    tokens = np.asarray(obs)[..., :token_count].argmax(-1)
    digits = np.stack(np.unravel_index(tokens, (3,) * factors), axis=-1)
    batch, length = tokens.shape
    masks = [(0,)] if factors == 1 else [(0,), (1,), (0, 1)]
    for mask in masks:
        for swap in (True, False):
            twin = digits.copy()
            for factor in mask:
                if swap:
                    twin[..., factor] = np.take([1, 0, 2], digits[..., factor])
                else:
                    shape = (batch, -(-length // REGIME_LENGTH))
                    regimes = rng.integers(3, size=shape).repeat(REGIME_LENGTH, axis=1)[:, :length]
                    u = rng.random((batch, length))
                    twin[..., factor] = (u[..., None] > emission[regimes].cumsum(-1)[..., :2]).sum(-1)
            edited = np.asarray(obs).copy()
            joint = np.ravel_multi_index(tuple(np.moveaxis(twin, -1, 0)), (3,) * factors)
            edited[..., :token_count] = np.eye(token_count, dtype=edited.dtype)[joint]
            yield edited


@partial(jax.jit, static_argnums=(0,))
def replay(spec, params, process, observations):
    """Replay full prefixes with the archived previous-action convention."""
    batch, length, _ = observations.shape
    token_count = process.n_tokens
    tokens = observations[..., :token_count].argmax(-1)
    prev_actions = observations[..., token_count:].argmax(-1)
    initial = jax.vmap(env.belief_reset, (None, 0))(process, tokens[:, 0])

    def step(state, data):
        carry, belief = state
        t, obs, token, action = data
        belief = jax.lax.cond(t == 0, lambda: belief, lambda: jax.vmap(
            env.belief_update, (None, 0, 0, 0))(process, belief, action, token))
        _, sites, carry = forward(spec, params, carry, obs, t)
        return (carry, belief), (sites[:, SITE], belief)

    _, result = jax.lax.scan(step, (models.init_carry(spec, batch), initial),
                            (jnp.arange(length), observations.swapaxes(0, 1), tokens.T, prev_actions.T))
    return tuple(x.swapaxes(0, 1) for x in result)


def rows(array, warmup, count):
    a = np.asarray(array)[:, warmup:]
    return a.reshape(-1, *a.shape[2:])[:count]


def paired_rows(spec, params, process, obs, h, belief, emission, factors, seed, warmup, count):
    pairs = []
    for edited in edit_histories(obs, emission, factors, seed):
        h_twin, b_twin = replay(spec, params, process, edited)
        pairs.append((rows(h_twin, warmup, count), rows(b_twin, warmup, count)))
    return (np.tile(h, (len(pairs), 1)), np.tile(belief, (len(pairs), 1)),
            np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]))


def fit_write(h, b, twin_h, twin_b):
    """Regress opposite signed displacements with each partner's target belief."""
    x = np.concatenate((np.column_stack((h, twin_b)), np.column_stack((twin_h, b))))
    dh = twin_h.astype(np.float64) - h
    coef, bias = affine(x, np.concatenate((dh, -dh)))
    return dict(residual=coef[:h.shape[1]], target=coef[h.shape[1]:], offset=bias)


def affine(x, y):
    coef = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=RCOND)[0]
    return coef[:-1], coef[-1]


def displacement(h, goal, write):
    return h @ write["residual"] + goal @ write["target"] + write["offset"]


def scaled_shift(x, goal, write, strength, use_null, direction):
    shift = displacement(x, goal, write)
    null = jnp.linalg.norm(shift, axis=-1, keepdims=True) * direction
    return strength * jnp.where(use_null, null, shift)


def specificity(write, train, test, readout_targets):
    x, b, twin, target = test
    train_h, train_b, train_twin, train_target = train
    probe, _ = affine(np.concatenate((train_h, train_twin)),
                      np.concatenate((readout_targets(train_b), readout_targets(train_target))))
    shift = displacement(x, target, write)
    true_delta = readout_targets(target) - readout_targets(b)
    dh = twin - x
    predicted = shift @ probe
    error = np.mean((predicted - true_delta) ** 2, axis=0)
    coordinate_energy = np.mean(true_delta ** 2, axis=0)
    preserved = np.abs(true_delta) < .02
    return dict(
        test_pairs=len(x),
        activation_displacement_r2_about_zero=float(1 - np.sum((shift - dh) ** 2) / np.sum(dh ** 2)),
        belief_displacement_rmse=np.sqrt(error).tolist(),
        belief_displacement_rms_target=np.sqrt(coordinate_energy).tolist(),
        preserved_coordinate_mean_abs_shift=[
            float(np.abs(p[m]).mean()) if m.any() else None for p, m in zip(predicted.T, preserved.T)
        ],
    )


def support(train_b, train_target, b, desired):
    tree = cKDTree(np.concatenate((np.column_stack((train_b, train_target)),
                                   np.column_stack((train_target, train_b)))))
    distance, _ = tree.query(np.column_stack((b, desired)))
    return dict(nearest_pair_distance_p50_p90=np.quantile(distance, [.5, .9]).tolist(),
                fraction_within_point1=float(np.mean(distance <= .1)), rows=len(distance))
