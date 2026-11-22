# RockSample manuscript protocol extracted from alex-rl-experiments@372b3f886ad4b69e1fcdfb15f6d9350ec4a90568.
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm


def affine(x, y):
    design = np.column_stack((x, np.ones(len(x))))
    coef = np.linalg.lstsq(design, y, rcond=1e-6)[0]
    return coef[:-1], coef[-1]


def nuisance_features(rows, env):
    n, horizon = rows["valid"].shape
    position = rows["position"]
    cell = position[:, :, 0] + env.n * position[:, :, 1]
    time = np.broadcast_to(np.arange(horizon), (n, horizon))
    previous = np.full((n, horizon), env.num_actions)
    previous[:, 1:] = rows["action"][:, :-1]
    symbol = rows["obs"][:, :, 2 + 2*env.k:2 + 2*env.k + rs.NUM_SYMBOLS]
    picked = ((rows["action"] == 4)[:, :, None] & rows["valid"][:, :, None] &
              np.all(position[:, :, None] == np.asarray(env.rocks), axis=-1))
    sampled = np.zeros_like(picked)
    sampled[:, 1:] = np.maximum.accumulate(picked[:, :-1], axis=1)
    features = np.concatenate((np.eye(env.n**2)[cell][:, :, :-1],
        np.eye(env.episode_length)[time][:, :, 1:],
        np.eye(env.num_actions + 1)[previous][:, :, :env.num_actions],
        symbol[:, :, :-1], sampled), axis=-1)
    return features[rows["valid"]]


def nuisance_encoder(beliefs, nuisance, activations):
    design = np.column_stack((beliefs, nuisance))
    coefficients, _ = affine(design, activations)
    return coefficients[:beliefs.shape[1]]


@partial(jax.jit, static_argnames=("spec",))
def replay(spec, params, observations):
    def one(sequence):
        def advance(cache, obs):
            sites = []
            def capture(site, x):
                sites.append(x.reshape(-1))
                return x
            _, cache = tm.encode_step(spec, params, cache, obs,
                                      residual_transform=capture)
            return cache, jnp.stack(sites)
        return jax.lax.scan(advance, tm.empty_cache(spec), sequence)[1]
    return jax.vmap(one)(observations)
