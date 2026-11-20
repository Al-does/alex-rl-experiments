"""Closed-loop comparisons using the historical environment RNG schedule."""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import replay
from experiments.algorithm_generality_probes_eps02_2026_11 import paper_probe, steer_core as old
from experiments.belief_quotient_history_jax_2026_11 import env, models

SUFFIX = 8
MODES = {"mirror": 1, "rho_flip": 2, "rho_point": 3}
METHODS = {"intact": 0, "random": 1, "eq4": 2, "encoder_exact": 3, "encoder_decoded": 4, "reference": 5}


def construct(process, belief, observations, steps, goal, mode):
    """Greedy coordinate descent: retain the original or flip C status at each slot."""
    tokens = observations[..., :3].argmax(-1)
    actions = observations[..., 3:].argmax(-1)
    init = jnp.broadcast_to(process.initial, belief.shape)

    def refilter(ts):
        def body(b, args):
            token, action, t = args
            prior = jnp.einsum("bi,bij->bj", b, process.transitions[action])
            prior = jnp.where(t == 0, init, prior)
            p = prior * process.emission[:, token].T
            return jnp.where(t >= 0, p / p.sum(-1, keepdims=True), b), None
        return jax.lax.scan(body, belief, (ts.T, actions.T, steps))[0]

    def search(i, candidate):
        j = SUFFIX - 1 - i
        tok = tokens[:, j]
        options = (candidate, candidate.at[:, j].set(jnp.where(tok == 2, 0, 2)),
                   candidate.at[:, j].set(jnp.where(tok == 2, 1, 2)))
        filtered = jax.vmap(refilter)(jnp.stack(options))
        rho_error = jnp.abs(filtered[..., 2] - goal[:, 2])
        split_error = jnp.abs((filtered[..., 0] - filtered[..., 1]) - (goal[:, 0] - goal[:, 1]))
        loss = jnp.where(mode == 1, rho_error + split_error, rho_error + 1e-5 * split_error)
        winner = loss.argmin(0)
        return jnp.where(steps[j] >= 0, jnp.stack(options)[winner, jnp.arange(len(belief))], candidate)

    tokens = jax.lax.fori_loop(0, SUFFIX, search, tokens)
    obs = jnp.concatenate((jax.nn.one_hot(tokens, 3), observations[..., 3:]), -1)
    return obs, refilter(tokens)


def instrument(spec, params, carry, obs, t, site, shift):
    batch = len(obs)
    slot = t % (spec.context_len + 1)
    valid = jnp.arange(spec.context_len + 1) <= t
    x = models._dense(params["input"], obs)
    keys, values = [], []
    for layer, p in enumerate(params["blocks"]):
        q, k, v = jnp.split(models._dense(p["qkv"], models._layer_norm(p["ln1"], x)), 3, -1)
        shape = (batch, spec.n_heads, spec.head_dim)
        q, k = (models._rope(y.reshape(shape), t) for y in (q, k))
        kc = carry["k"][layer].at[:, slot].set(k)
        vc = carry["v"][layer].at[:, slot].set(v.reshape(shape))
        scores = jnp.einsum("bhd,bshd->bhs", q, kc) / math.sqrt(spec.head_dim)
        weights = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), -1)
        attended = jnp.einsum("bhs,bshd->bhd", weights, vc).reshape(batch, spec.width)
        x = x + models._dense(p["proj"], attended)
        x = x + models._mlp(p["mlp"], models._layer_norm(p["ln2"], x))
        if layer == site:
            x, norms = shift(x)
        keys.append(kc)
        values.append(vc)
    x = models._layer_norm(params["final_norm"], x)
    if site == 4:
        x, norms = shift(x)
    return models._dense(params["policy"], x), {"k": jnp.stack(keys), "v": jnp.stack(values)}, norms


@partial(jax.jit, static_argnames=("spec", "site", "family", "episodes", "length", "capture"))
def rollout(spec, process, coarse, params, maps, encoding, key, episodes, length,
            site, family, mode, method, strength, capture=False):
    mean, pc, coef, c_weight, c_bias = encoding
    init_c, coarse_e, coarse_t = coarse
    reset_key, key = jax.random.split(key)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    b = jax.vmap(env.belief_reset, (None, 0))(process, state.token)
    c = init_c * coarse_e[:, (state.token == 2).astype(int)].T
    c /= c.sum(-1, keepdims=True)
    cache = models.init_carry(spec, episodes)
    buffer = jnp.zeros((episodes, SUFFIX, spec.obs_dim), dtype=jnp.float32)
    beliefs = jnp.broadcast_to(process.initial, (episodes, SUFFIX, 3))
    saved_logits = jnp.zeros((episodes, SUFFIX, 3), dtype=jnp.float32)

    def body(carry, t):
        state, b, c, cache, shadow_cache, delayed, buffer, beliefs, saved_logits, key = carry
        key, ak, ek = jax.random.split(key, 3)
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        shadow, shadow_cache, _, _ = old.instrumented_step(spec, params, shadow_cache, obs, t, maps)
        goal = old.target(b, mode)
        if capture:
            need_ref = True
        else:
            need_ref = (method == METHODS["reference"])

        def ref():
            steps = t + jnp.arange(1 - SUFFIX, 1)
            seq = jnp.concatenate((buffer[:, 1:], obs[:, None]), 1)
            alt, alt_b = construct(process, beliefs[:, 0], seq, steps, goal, mode)
            def replay_step(cache, args):
                index, observation = args
                def run_step():
                    z, c, h, _ = old.instrumented_step(spec, params, cache, observation, index, maps)
                    return c, (z, h[:, site])
                return jax.lax.cond(index >= 0, run_step,
                                    lambda: (cache, (jnp.zeros_like(shadow), jnp.zeros((episodes, spec.width)))))

            _, (zs, hs) = jax.lax.scan(replay_step, delayed, (steps, alt.swapaxes(0, 1)))
            historical_actions = seq[:, 1:, 3:].argmax(-1)
            lp = jax.nn.log_softmax(saved_logits[:, 1:] / 1.5)
            lq = jax.nn.log_softmax(zs[:-1].swapaxes(0, 1) / 1.5)
            mask = steps[:-1] >= 0
            drop = jnp.take_along_axis(lp - lq, historical_actions[..., None], -1)[..., 0]
            drop = (drop * mask).sum(-1) / jnp.maximum(mask.sum(), 1)
            return zs[-1], hs[-1], alt_b, drop

        z_ref, h_ref, b_ref, drop = jax.lax.cond(
            need_ref, ref, lambda: (shadow, jnp.zeros((episodes, spec.width)), b, jnp.zeros(episodes)))
        old_index = t - (SUFFIX - 1)
        delayed = jax.lax.cond(old_index >= 0,
                               lambda: old.instrumented_step(spec, params, delayed, buffer[:, 1], old_index, maps)[1],
                               lambda: delayed)

        def shift(h):
            def eq4():
                return old.ladder_displacement(h, maps, site, mode) @ maps["encoder"][site]

            def encode(decoded):
                bp = old.decode(h, maps, site) if decoded else b
                if decoded:
                    bp = jnp.clip(bp, 0., 1.)
                    bp /= bp.sum(-1, keepdims=True)
                cp = jnp.clip(h @ c_weight + c_bias, 0., 1.) if decoded else c[:, 1]
                desired = old.target(bp, mode)
                cp_target = jnp.where(mode == 1, cp, desired[:, 2])
                x = jnp.concatenate((bp, cp[:, None]), -1)
                y = jnp.concatenate((desired, cp_target[:, None]), -1)
                design = replay.design(x, y, h, jnp.zeros_like(bp), family, mean, pc, jnp)
                return design @ coef

            delta = jax.lax.switch(method, (lambda: jnp.zeros_like(h), lambda: jnp.zeros_like(h),
                                            eq4, lambda: encode(False), lambda: encode(True), lambda: h_ref - h))
            delta *= strength
            a, d = jnp.linalg.norm(h, axis=-1), jnp.linalg.norm(delta, axis=-1)
            return h + delta, jnp.stack((d, a, d / jnp.maximum(a, 1e-12)), -1)

        z, cache, norms = instrument(spec, params, cache, obs, t, site, shift)
        action = jnp.where(method == METHODS["random"], jax.random.randint(ak, (episodes,), 0, 3), z.argmax(-1))
        nxt, reward, _ = jax.vmap(env.step, (None, 0, 0, 0))(process, jax.random.split(ek, episodes), state, action)
        new_b = jax.vmap(env.belief_update, (None, 0, 0, 0))(process, b, action, nxt.token)
        new_c = jnp.einsum("bi,bij->bj", c, coarse_t[action]) * coarse_e[:, (nxt.token == 2).astype(int)].T
        new_c /= new_c.sum(-1, keepdims=True)
        buffer = jnp.concatenate((buffer[:, 1:], obs[:, None]), 1)
        beliefs = jnp.concatenate((beliefs[:, 1:], b[:, None]), 1)
        saved_logits = jnp.concatenate((saved_logits[:, 1:], shadow[:, None]), 1)
        record = dict(reward=reward, action=action, state=state.state, belief=b,
                      changed=action != shadow.argmax(-1), norms=norms)
        if capture:
            record.update(base=shadow, replay=z_ref, written=z, drop=drop, rho_error=jnp.abs(b_ref[:, 2] - goal[:, 2]),
                          delta_error=jnp.abs((b_ref[:, 0] - b_ref[:, 1]) - (goal[:, 0] - goal[:, 1])))
        return (nxt, new_b, new_c, cache, shadow_cache, delayed, buffer, beliefs, saved_logits, key), record

    first = (state, b, c, cache, cache, cache, buffer, beliefs, saved_logits, key)
    return jax.lax.scan(body, first, jnp.arange(length))[1]


def compare(spec, process, params, fit, encoders, patch_selection, old_records, smoke=False):
    maps = old.fit_maps({k: v.swapaxes(0, 1) for k, v in fit.items()}, 12000)
    coarse = tuple(jnp.asarray(x, dtype=jnp.float32) for x in paper_probe.lumped_model(process._asdict()))
    c = replay.filtered(process, fit["obs"], 2)[..., 3][:, replay.WARMUP:].reshape(-1)
    def choice(site):
        effect = [row[site]["latest"]["all"] for key, row in patch_selection.items() if key != "0_0"]
        usable = [x for x in effect if x["fidelity"] is not None]
        return np.average([x["fidelity"] for x in usable], weights=[x["replay_tv"] for x in usable])

    supported = max(range(5), key=choice)
    out = dict(supported_site=supported, selection_latest_fidelity=[float(choice(i)) for i in range(5)], sites={})
    eval_episodes, length = (2, 64) if smoke else (32, 1024)
    tune_episodes = 2 if smoke else 16
    for site in sorted({3, supported}):
        family, mean, pc, coef = encoders[site]
        w, bias = old.affine(fit["h"][:, replay.WARMUP:, site].reshape(-1, spec.width), c)
        encoding = tuple(jnp.asarray(x, dtype=jnp.float32) for x in (mean, pc, coef, w, bias))

        def run(seed, episodes, name, strength, mode, capture=False):
            return rollout(spec, process, coarse, params, maps, encoding, jax.random.PRNGKey(seed),
                           episodes, length, site, family, MODES[mode], METHODS[name], strength, capture)

        random = float(run(1042, tune_episodes, "random", 0., "rho_flip")["reward"].mean())
        data = {}
        for control in ("intact", "random"):
            data[control] = old.metrics(run(2042, eval_episodes, control, 0., "rho_flip"))
        for mode in MODES:
            data[mode] = {}
            for method in ("eq4", "encoder_exact", "encoder_decoded", "reference"):
                if mode == "mirror":
                    strength, curve = 1., None
                elif method == "eq4":
                    row, = [r for r in old_records["confirmation"] if r["site"] == 3 and r["condition"] == f"selected_{mode}"]
                    strength, curve = row["strength"], None
                else:
                    grid = (0., 1.) if smoke else old.LAMBDAS
                    curve = [float(run(1042, tune_episodes, method, s, mode)["reward"].mean()) for s in grid]
                    strength = next((s for s, occ in zip(grid, curve) if occ <= random), grid[int(np.argmin(curve))])
                rows = {}
                for lam in sorted({strength, 1.}):
                    r = {k: np.asarray(v) for k, v in run(2042, eval_episodes, method, lam, mode, True).items()}
                    m = old.metrics(r)
                    m.update(fidelity=replay.fidelity(r["base"], r["replay"], r["written"], r["drop"]),
                             replay_rho_mae=float(r["rho_error"].mean()), replay_delta_mae=float(r["delta_error"].mean()))
                    for label, mask in (("low_drop", r["drop"] < replay.DROP), ("high_drop", r["drop"] >= replay.DROP)):
                        if mask.any():
                            m["fidelity"][label].update(occupancy=float(r["reward"][mask].mean()),
                                                        action_change=float(r["changed"][mask].mean()),
                                                        rho_mae=float(r["rho_error"][mask].mean()))
                    rows[str(lam)] = m
                data[mode][method] = dict(strength=strength, tune_occupancy=curve, confirmation=rows)
        out["sites"][str(site)] = data
    return out
