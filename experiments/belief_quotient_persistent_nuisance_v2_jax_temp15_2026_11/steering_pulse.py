"""Single-write open-loop branches and matched exact-filter predictions."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.environment import action_transitions, EMISSION_MATRIX
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import instrumented_step, pulse_branch
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import target


@partial(jax.jit, static_argnums=(0, 5))
def snapshots(spec, params, maps, observations, times, count):
    cache = models.init_carry(spec, observations.shape[1])
    saved = jax.tree_util.tree_map(lambda x: jnp.zeros((count,)+x.shape), cache)
    def body(carry, item):
        cache, saved = carry
        t, obs = item
        saved = jax.tree_util.tree_map(lambda a, c: jax.lax.fori_loop(0, count, lambda i, a: a.at[i].set(jnp.where(t == times[i], c, a[i])), a), saved, cache)
        _, cache, _, _ = instrumented_step(spec, params, cache, obs, t, maps)
        return (cache, saved), None
    (_, saved), _ = jax.lax.scan(body, (cache, saved), (jnp.arange(len(observations)), observations))
    return saved


def decay(curve):
    y = np.asarray(curve)
    k = np.arange(len(y))
    valid = (k > 0) & (y > max(1e-12, y.max()*1e-8))
    if valid.sum() < 3:
        return dict(multiplier=None, r2=None, n_lags=int(valid.sum()))
    slope, intercept = np.polyfit(k[valid], np.log(y[valid]), 1)
    truth = np.log(y[valid])
    variance = ((truth-truth.mean())**2).sum()
    return dict(multiplier=float(np.exp(slope)), r2=float(1-((truth-intercept-slope*k[valid])**2).sum()/variance) if variance else None, n_lags=int(valid.sum()))


def coordinates(b):
    return np.stack((b[..., 2], b[..., 0]-b[..., 1]), -1)


def exact_trace(b, observations, variant, mode, coarse=False):
    b = np.asarray(b, float).copy()
    perturbed = target(b, mode)
    transition = action_transitions(variant)
    differences = [coordinates(perturbed)-coordinates(b)]
    for obs in observations[1:]:
        action = obs[:, 3:].argmax(-1)
        token = obs[:, :3].argmax(-1)
        if coarse:
            def update(x):
                rho = x[:, 2]
                r = (1-rho)*transition[action, 0, 2]+rho*transition[action, 2, 2]
                rho = np.where(token == 2, r/(r+.85*(1-r)), 0)
                return np.stack(((1-rho)/2, (1-rho)/2, rho), -1)
        else:
            def update(x):
                x = np.einsum('bs,bsj->bj', x, transition[action])*EMISSION_MATRIX[:, token].T
                return x/x.sum(-1, keepdims=True)
        b, perturbed = update(b), update(perturbed)
        differences.append(coordinates(perturbed)-coordinates(b))
    return np.asarray(differences)


def summarize(difference):
    signed, absolute = difference.mean(1), np.abs(difference).mean(1)
    return dict(signed=signed.tolist(), absolute=absolute.tolist(),
                decay={name: decay(absolute[:, i]) for i, name in enumerate(('rho', 'delta'))})


def run_pulse(spec, params, maps, records, variant, times=(256, 512, 768), horizon=128):
    observations = records['obs']
    saved = snapshots(spec, params, maps, observations, jnp.asarray(times), len(times))
    results = {}
    for mode, mode_index in (('mirror', 1), ('rho_flip', 2)):
        exact, coarse = [], []
        for t in times:
            obs = np.asarray(observations[t:t+horizon+1])
            b = np.asarray(records['belief'][t])
            exact.append(exact_trace(b, obs, variant, mode))
            if variant == 2:
                coarse.append(exact_trace(b, obs, variant, mode, coarse=True))
        for site in (0, 1, 2):
            diffs, tvs, disagreements = [], [], []
            for index, t in enumerate(times):
                cache = jax.tree_util.tree_map(lambda x: x[index], saved)
                obs = observations[t:t+horizon+1]
                _, intact = pulse_branch(spec, params, maps, cache, obs, t, site, mode_index, 0.)
                _, branch = pulse_branch(spec, params, maps, cache, obs, t, site, mode_index, 1.)
                diffs.append(coordinates(np.asarray(branch[0]))-coordinates(np.asarray(intact[0])))
                tvs.append(np.abs(np.asarray(branch[1])-np.asarray(intact[1])).sum(-1)/2)
                disagreements.append(np.asarray(branch[2]) != np.asarray(intact[2]))
            result = dict(measured=summarize(np.concatenate(diffs, 1)), full_filter=summarize(np.concatenate(exact, 1)),
                          tv=np.concatenate(tvs, 1).mean(1).tolist(), action_disagreement=np.concatenate(disagreements, 1).mean(1).tolist())
            if coarse:
                result['quotient_filter'] = summarize(np.concatenate(coarse, 1))
                result['quotient_filter']['delta_not_defined'] = True
            results[f'{mode}_block{site+1}'] = result
    return dict(seed=3042, times=list(times), horizon=horizon, conditions=results)
