"""Reward-State Quotient twin steering at the paper's primary site."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.algorithm_generality_probes_eps02_2026_11 import steer_core as old
from experiments.algorithm_generality_probes_eps02_2026_11.steer_moving_base import forward
from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment

from . import twins


def logits_at_block4(params, h):
    return models._dense(params["policy"], models._layer_norm(params["final_norm"], h))


@partial(jax.jit, static_argnums=(0, 5, 6))
def rollout(spec, process, params, maps, key, episodes, length, mode, strength, use_null=False,
            random_actions=False):
    reset_key, key = jax.random.split(key)
    initial = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))

    def body(state, t):
        env_state, carry, key = state
        obs = jax.vmap(env.observe, (None, 0))(process, env_state)
        intact, sites, carry = forward(spec, params, carry, obs, t)
        h = sites[:, twins.SITE]
        goal = old.target(old.decode(h, maps, twins.SITE), mode)
        delta = twins.scaled_shift(h, goal[:, :2], maps["write"], strength, use_null,
                                  maps["orthogonal"][twins.SITE])
        logits = logits_at_block4(params, h + delta)
        key, ak, ek = jax.random.split(key, 3)
        action = jnp.where(random_actions, jax.random.randint(ak, (episodes,), 0, 3), logits.argmax(-1))
        next_state, reward, _ = jax.vmap(env.step, (None, 0, 0, 0))(
            process, jax.random.split(ek, episodes), env_state, action)
        norm, hnorm = jnp.linalg.norm(delta, axis=-1), jnp.linalg.norm(h, axis=-1)
        norm = jnp.where((strength == 0) | random_actions, 0., norm)
        hnorm = jnp.where((strength == 0) | random_actions, 0., hnorm)
        return (next_state, carry, key), dict(
            reward=reward, state=env_state.state, action=action, changed=action != intact.argmax(-1),
            norms=jnp.stack((norm, hnorm, norm / jnp.maximum(hnorm, 1e-12)), -1))

    return jax.lax.scan(body, (initial, models.init_carry(spec, episodes), key),
                        jnp.arange(length))[1]


def contrasts(b):
    return np.column_stack((1 - b.sum(-1), b[:, 0] - b[:, 1]))


def run(spec, process, params, baseline, smoke=False):
    rows, length, warmup = (256, 256, 8) if smoke else (12000, 1024, 8)
    fit_episodes = -(-rows // (length - 128))
    tune_episodes, confirm_episodes = (2, 2) if smoke else (16, 32)
    empty = old.empty_maps(spec.width)

    def collect(seed):
        return old.rollout(spec, process, params, empty, jax.random.PRNGKey(seed), fit_episodes, length, capture=True)

    def make_pairs(records, seed):
        h = twins.rows(records["h"].swapaxes(0, 1)[:, :, twins.SITE], warmup, rows)
        b = twins.rows(records["belief"].swapaxes(0, 1), warmup, rows)
        x, y, tx, ty = twins.paired_rows(
            spec, params, process, records["obs"].swapaxes(0, 1), h, b,
            environment.emission_matrix(), 1, seed, warmup, rows)
        return x, y[:, :2], tx, ty[:, :2]

    fit, heldout = collect(twins.FIT_SEED), collect(twins.TEST_SEED)
    maps = old.fit_maps(fit, rows)
    train, test = make_pairs(fit, twins.FIT_SEED), make_pairs(heldout, twins.TEST_SEED)
    writer = twins.fit_write(*train)
    maps["write"] = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), writer)
    x, b, _, _ = test
    x, b = x[:rows:12], b[:rows:12]
    decoded = old.decode(jnp.asarray(x), maps, twins.SITE)
    diagnostics = dict(
        coordinates=["rho", "delta"], paired_write=twins.specificity(writer, train, test, contrasts),
        goal_support={name: twins.support(train[1], train[3], b, np.asarray(old.target(decoded, mode))[:, :2])
                      for mode, name in enumerate(old.MODES)},
        old_probe_r2=old.probe_scores(heldout, maps, rows, warmup)[twins.SITE])
    print(f'ladder diagnostic seed {baseline["seed"]}: {diagnostics}', flush=True)

    def evaluate(seed, episodes, mode=0, strength=0., null=False, random=False):
        record = rollout(spec, process, params, maps, jax.random.PRNGKey(seed), episodes, length,
                         jnp.int32(mode), jnp.float32(strength), null, random)
        result = old.metrics(record)
        del result["episode_occupancy"]
        return result

    random_tune = evaluate(twins.TUNE_SEED, tune_episodes, random=True)["occupancy"]
    intact = evaluate(twins.CONFIRM_SEED, confirm_episodes)
    random = evaluate(twins.CONFIRM_SEED, confirm_episodes, random=True)
    comparisons = {}
    for mode, name in enumerate(old.MODES):
        grid = (0., 1.) if smoke else old.LAMBDAS
        sweep = {l: evaluate(twins.TUNE_SEED, tune_episodes, mode, l)["occupancy"] for l in grid}
        eligible = [l for l in grid if sweep[l] <= random_tune]
        strength = min(eligible) if eligible else min(grid, key=sweep.get)
        comparisons[name] = dict(
            selected_strength=strength, tuning_occupancy=sweep,
            selected=evaluate(twins.CONFIRM_SEED, confirm_episodes, mode, strength),
            full_strength=evaluate(twins.CONFIRM_SEED, confirm_episodes, mode, 1.),
            matched_null=evaluate(twins.CONFIRM_SEED, confirm_episodes, mode, strength, null=True),
            full_strength_matched_null=evaluate(twins.CONFIRM_SEED, confirm_episodes, mode, 1., null=True))
    return dict(old=baseline, diagnostics=diagnostics, intact=intact, random=random, comparisons=comparisons,
                protocol=dict(fit_rows=rows, length=length, warmup=warmup, tune_episodes=tune_episodes,
                              confirmation_episodes=confirm_episodes, tune_random_occupancy=random_tune))
