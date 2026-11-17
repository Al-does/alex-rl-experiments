"""Paper pulse protocol adapted to the output and carry of every GRU layer."""
from functools import partial

from .reference import core, pulse

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import env, models


def empty_maps(spec):
    return {"decoder": jnp.zeros((spec.n_layers, spec.width, 2)),
            "bias": jnp.zeros((spec.n_layers, 2)),
            "encoder": jnp.zeros((spec.n_layers, 2, spec.width))}


def step(spec, params, carry, obs, t, maps, site=-1, mode=0, strength=0.):
    x = models._dense(params["input"], obs)
    hidden = []
    for layer, block in enumerate(params["blocks"]):
        x = models._gru_cell(block, x, carry[layer])

        def install(h):
            displacement = core.ladder_displacement(h, maps, layer, mode)
            shift = jnp.matmul(displacement, maps["encoder"][layer], precision=jax.lax.Precision.HIGHEST)
            return h + strength * shift

        x = jax.lax.cond(site == layer, install, lambda h: h, x)
        hidden.append(x)
    hidden = jnp.stack(hidden)
    logits, _, _ = models._head(params, x)
    return logits, hidden, jnp.swapaxes(hidden, 0, 1)


@partial(jax.jit, static_argnums=(0, 5, 6))
def rollout(spec, process, params, maps, key, episodes, length):
    reset_key, key = jax.random.split(key)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    belief = jax.vmap(env.belief_reset, (None, 0))(process, state.token)
    hidden = models.init_carry(spec, episodes)

    def body(carry, t):
        state, belief, hidden, key = carry
        key, _, ek = jax.random.split(key, 3)
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        logits, hidden, h = step(spec, params, hidden, obs, t, maps)
        action = jnp.argmax(logits, -1)
        new_state, reward, _ = jax.vmap(env.step, (None, 0, 0, 0))(
            process, jax.random.split(ek, episodes), state, action)
        new_belief = jax.vmap(env.belief_update, (None, 0, 0, 0))(
            process, belief, action, new_state.token)
        return (new_state, new_belief, hidden, key), {"h": h, "obs": obs,
                "belief": belief, "action": action, "reward": reward}

    return jax.lax.scan(body, (state, belief, hidden, key), jnp.arange(length))[1]


def retained(records, rows, warmup=8):
    h = np.asarray(records["h"])[warmup:]
    b = np.asarray(records["belief"])[warmup:]
    if not h.size or rows <= 0:
        raise ValueError("nonempty retained histories and positive row budget required")
    h = h.transpose(1, 0, 2, 3).reshape(-1, *h.shape[2:])[:rows]
    b = b.transpose(1, 0, 2).reshape(-1, 3)[:rows]
    if len(h) != rows:
        raise ValueError("insufficient retained rows")
    return h, b


def fit_maps(records, rows):
    h, b = retained(records, rows)
    decoder, bias, encoder = [], [], []
    for site in range(h.shape[1]):
        d, c = core.affine(h[:, site], b[:, :2])
        e, _ = core.affine(b[:, :2], h[:, site])
        decoder.append(d)
        bias.append(c)
        encoder.append(e)
    return {key: jnp.asarray(np.stack(value), dtype=jnp.float32)
            for key, value in (("decoder", decoder), ("bias", bias), ("encoder", encoder))}


def probe_scores(records, maps, rows):
    h, b = retained(records, rows)
    true = np.column_stack((b[:, 2], b[:, 0] - b[:, 1]))
    variance = ((true - true.mean(0)) ** 2).sum(0)
    scores = []
    for site in range(h.shape[1]):
        prediction = np.asarray(core.decode(jnp.asarray(h[:, site]), maps, site))
        pred = np.column_stack((prediction[:, 2], prediction[:, 0] - prediction[:, 1]))
        error = np.divide(((pred - true) ** 2).sum(0), variance,
                          out=np.full(2, np.nan), where=variance > 0)
        scores.append({k: None if not np.isfinite(x) else float(1-x)
                       for k, x in zip(("rho", "delta"), error)})
    return scores


@partial(jax.jit, static_argnums=(0, 5))
def snapshots(spec, params, maps, observations, times, count):
    hidden = models.init_carry(spec, observations.shape[1])
    saved = jnp.zeros((count,) + hidden.shape)

    def body(carry, item):
        hidden, saved = carry
        t, obs = item
        saved = jax.lax.fori_loop(0, count, lambda i, a: a.at[i].set(
            jnp.where(t == times[i], hidden, a[i])), saved)
        _, hidden, _ = step(spec, params, hidden, obs, t, maps)
        return (hidden, saved), None

    return jax.lax.scan(body, (hidden, saved), (jnp.arange(len(observations)), observations))[0][1]


@partial(jax.jit, static_argnums=(0,))
def branch(spec, params, maps, hidden, observations, start, site, mode, strength):
    def body(hidden, item):
        lag, obs = item
        _, hidden, h = step(spec, params, hidden, obs, start + lag, maps, site, mode,
                            jnp.where(lag == 0, strength, 0.))
        return hidden, core.decode(h[:, -1], maps, spec.n_layers - 1)
    return jax.lax.scan(body, hidden, (jnp.arange(len(observations)), observations))


def run_pulse(spec, params, maps, records, variant, environment,
              times=(256, 512, 768), horizon=128):
    observations = records["obs"]
    if not times or min(times) < 0 or max(times) + horizon >= len(observations):
        raise ValueError("pulse times and horizon must fit the complete recorded episode")
    saved = snapshots(spec, params, maps, observations, jnp.asarray(times), len(times))
    results = {}
    for mode, mode_index in (("mirror", 1), ("rho_flip", 2)):
        exact = [pulse.exact_trace(np.asarray(records["belief"][t]),
                 np.asarray(observations[t:t+horizon+1]), variant, mode,
                 environment=environment) for t in times]
        for site in range(spec.n_layers):
            differences = []
            for i, t in enumerate(times):
                obs = observations[t:t+horizon+1]
                _, intact = branch(spec, params, maps, saved[i], obs, t, site, mode_index, 0.)
                _, written = branch(spec, params, maps, saved[i], obs, t, site, mode_index, 1.)
                differences.append(pulse.coordinates(np.asarray(written)) - pulse.coordinates(np.asarray(intact)))
            results[f"{mode}_layer{site+1}"] = {
                "measured": pulse.summarize(np.concatenate(differences, 1)),
                "full_filter": pulse.summarize(np.concatenate(exact, 1))}
    return {"seed": 3042, "times": list(times), "horizon": horizon, "conditions": results}
