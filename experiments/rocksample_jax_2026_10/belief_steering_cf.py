"""Counterfactual-twin belief steering of the r7 RockSample policies.

A twin history repeats an episode's positions and actions exactly and flips only
the Good/Bad sensor symbols returned for one rock. With a uniform prior and a
symmetric sensor the Bayesian marginal of that rock becomes ``1 - m`` while every
other marginal and every observable nuisance is unchanged. Twin activation
differences therefore identify how the network writes one rock's belief without
regressing out nuisances. The same twin, run in parallel during a closed-loop
rollout, supplies interchange (activation-patching) references.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_probe as bp
from experiments.rocksample_jax_2026_10 import belief_steering as bs
from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm

SITES = 4
LOGIT_CLIP = 0.02
PROTO_MARGIN = 0.1
Z95 = 1.959963984540054


def symbol_index(env):
    return 2 + 2 * env.k


def check_index(env, rock):
    return symbol_index(env) + rs.NUM_SYMBOLS + rs.NUM_BASE_ACTIONS + rock


def flip_symbols(obs, gate, env):
    """Swap the Good and Bad symbol entries wherever ``gate`` is true."""
    s = symbol_index(env)
    xp = jnp if isinstance(obs, jax.Array) else np
    swapped = xp.concatenate((obs[..., :s], obs[..., s + 1:s + 2], obs[..., s:s + 1],
                              obs[..., s + 2:]), axis=-1)
    return xp.where(gate[..., None], swapped, obs)


def logit(m):
    m = np.clip(m, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(m) - np.log1p(-m)


def sampled_history(rows, env):
    position = rows["position"]
    picked = ((rows["action"] == 4)[:, :, None] & rows["valid"][:, :, None] &
              np.all(position[:, :, None] == np.asarray(env.rocks), axis=-1))
    sampled = np.zeros_like(picked)
    sampled[:, 1:] = np.maximum.accumulate(picked[:, :-1], axis=1)
    return sampled


def fit(spec, env, params, episodes, seed, *, ranks=(1, 2, 4, 8)):
    """Probe decoders plus twin-difference encoders and subspaces at every site."""
    rows = bp.collect(spec, env, params, params, episodes=episodes, key_seed=seed)
    data = bp.targets(rows, env)
    valid = rows["valid"]
    h = np.asarray(bs.replay(spec, params, rows["obs"]))[valid].astype(np.float64)
    m, relevant = data.marginals, data.relevant
    sampled = sampled_history(rows, env)
    nuisance = bs.nuisance_features(rows, env)
    targets = dict(prob=m, logit=logit(m))
    out = dict(rows=int(valid.sum()), decoder={}, offset={}, r2={}, encoder={}, basis={},
               diagnostics={})
    for name, y in targets.items():
        fits = [bs.affine(h[:, s], y) for s in range(SITES)]
        out["decoder"][name] = np.stack([f[0] for f in fits])
        out["offset"][name] = np.stack([f[1] for f in fits])
        out["encoder"][f"plain_{name}"] = np.stack(
            [bs.affine(y, h[:, s])[0] for s in range(SITES)])
        out["encoder"][f"nuisance_{name}"] = np.stack(
            [bs.nuisance_encoder(y, nuisance, h[:, s]) for s in range(SITES)])
    twin = {k: np.zeros((SITES, env.k, spec.d_model)) for k in
            ("twin_prob", "twin_logit", "twinbelief_prob", "twinbelief_logit", "twin_mean")}
    basis = np.zeros((SITES, env.k, max(ranks), spec.d_model))
    basis_all = np.zeros_like(basis)
    proto = np.zeros((2, SITES, env.k, spec.d_model))
    reflect = {b: np.zeros((SITES, env.k, len(ranks), spec.d_model + 1, spec.d_model))
               for b in ("pca", "pca_all")}
    mid = np.zeros((SITES, env.k, spec.d_model))
    diag = []
    for rock in range(env.k):
        check = rows["obs"][:, :, check_index(env, rock)] > .5
        gate = check & ~sampled[:, :, rock]
        hcf = np.asarray(bs.replay(spec, params, flip_symbols(rows["obs"], gate, env)))
        hcf = hcf[valid].astype(np.float64)
        dh = hcf - h
        rel = relevant[:, rock]
        for quality, low in ((0, True), (1, False)):
            real = rel & ((m[:, rock] < PROTO_MARGIN) if low else (m[:, rock] > 1 - PROTO_MARGIN))
            cf = rel & ((m[:, rock] > 1 - PROTO_MARGIN) if low else (m[:, rock] < PROTO_MARGIN))
            proto[quality, :, rock] = np.concatenate((h[real], hcf[cf])).mean(0)
        mid[:, rock] = (h[rel] + hcf[rel]).mean(0) / 2
        other = rel & ~check[valid]
        dm = (1 - 2 * m[:, rock])
        dl = -2 * targets["logit"][:, rock]
        row = dict(rock=rock, relevant=int(rel.sum()), check_tokens=int((rel & check[valid]).sum()))
        for s in range(SITES):
            for key, x, mask in (("twin_prob", dm, rel), ("twin_logit", dl, rel),
                                 ("twinbelief_prob", dm, other), ("twinbelief_logit", dl, other)):
                e = (x[mask] @ dh[mask, s]) / (x[mask] @ x[mask])
                twin[key][s, rock] = e
                resid = dh[mask, s] - np.outer(x[mask], e)
                row[f"{key}_r2_site{s}"] = float(1 - (resid**2).sum() / (dh[mask, s]**2).sum())
            sign = np.sign(dm[other])
            twin["twin_mean"][s, rock] = (sign[:, None] * dh[other, s]).mean(0)
            _, sv, vt = np.linalg.svd(dh[other, s], full_matrices=False)
            basis[s, rock] = vt[:max(ranks)]
            basis_all[s, rock] = np.linalg.svd(dh[rel, s], full_matrices=False)[2][:max(ranks)]
            x = np.concatenate((h[rel, s], hcf[rel, s]))
            design = np.column_stack((x, np.ones(len(x))))
            for b, u_all in (("pca", basis[s, rock]), ("pca_all", basis_all[s, rock])):
                for j, rank in enumerate(ranks):
                    u = u_all[:rank]
                    y = dh[rel, s] @ u.T
                    y = np.concatenate((y, -y))
                    coef = np.linalg.lstsq(design, y, rcond=1e-6)[0]
                    reflect[b][s, rock, j] = coef @ u
                    row[f"reflect_{b}{rank}_r2_site{s}"] = float(
                        1 - ((design @ coef - y)**2).sum() / (y**2).sum())
            row[f"proto_gap_site{s}"] = float(np.linalg.norm(
                basis[s, rock] @ (proto[1, s, rock] - proto[0, s, rock])))
            row[f"spectrum_site{s}"] = (sv[:8]**2 / (sv**2).sum()).tolist()
            row[f"twin_norm_site{s}"] = float(np.linalg.norm(dh[other, s], axis=1).mean())
            row[f"h_norm_site{s}"] = float(np.linalg.norm(h[:, s], axis=1).mean())
        diag.append(row)
    out["encoder"].update(twin)
    out["basis"] = basis
    out["basis_all"] = basis_all
    out["proto"] = proto
    out["reflect"] = reflect
    out["mid"] = mid
    out["ranks"] = tuple(ranks)
    out["diagnostics"] = diag
    for name, y in targets.items():
        pred = np.einsum("nsd,sdk->nsk", h, out["decoder"][name]) + out["offset"][name]
        out["r2"][name] = (1 - ((pred - y[:, None])**2).sum(0) /
                           ((y - y.mean(0))**2).sum(0)).tolist()
    return out


def calibrate(encoder, decoder):
    """Rows moved so each changes its own decoded coordinate by one and no other."""
    return np.stack([np.linalg.solve(e @ w, e) for e, w in zip(encoder, decoder)])


def intervention(fitted, rock, good, *, encoder=None, space="prob", sites=(), strength=1.,
                 add=0., patch=None, patch_sites=(), rank=0, gate_patch=(1., 1.),
                 gate_steer=(1., 1.), control=False, null_seed=0, target=None,
                 source="twin", basis_rock=None, proto_scale=1.,
                 reflect=None, reflect_sites=(), reflect_scale=1., mirror=False):
    """Array-only description of one intervention; every field has a fixed shape."""
    d_model = fitted["decoder"]["prob"].shape[1]
    w = fitted["decoder"][space]
    iv = dict(d=w[:, :, rock], b=fitted["offset"][space][:, rock],
              e=np.zeros((SITES, d_model)), lam=np.zeros(SITES), add=np.zeros((SITES, d_model)),
              patch=np.zeros((SITES, d_model, d_model)), gate_patch=np.asarray(gate_patch),
              gate_steer=np.asarray(gate_steer), control=np.asarray(control),
              null=np.zeros((SITES, d_model)), use_proto=np.asarray(source == "proto"))
    other = fitted["proto"][int(bool(good)), :, rock]
    iv["proto"] = other + proto_scale * (fitted["proto"][int(not good), :, rock] - other)
    iv["lin"] = np.zeros((SITES, d_model, d_model))
    iv["off"] = np.zeros((SITES, d_model))
    iv["ro_w"], iv["ro_b"] = w[SITES - 1], fitted["offset"][space][SITES - 1]
    for s in reflect_sites:
        if mirror:
            u = fitted["basis" if reflect == "pca" else "basis_all"][s, rock, :rank]
            p = u.T @ u
            iv["lin"][s], iv["off"][s] = -2 * reflect_scale * p, 2 * reflect_scale * fitted["mid"][s, rock] @ p
        else:
            m = fitted["reflect"][reflect][s, rock, fitted["ranks"].index(rank)]
            iv["lin"][s], iv["off"][s] = reflect_scale * m[:-1], reflect_scale * m[-1]
    if target is None:
        target = (0. if good else 1.) if space == "prob" else float(
            logit(np.asarray(0. if good else 1.)))
    iv["target"] = np.asarray(target)
    if encoder is not None:
        e = calibrate(fitted["encoder"][encoder], w)[:, rock]
        iv["e"] = e
        sign = -1. if good else 1.
        unit = (1. if space == "prob" else float(logit(np.asarray(1.))))
        for s in sites:
            iv["lam"][s] = strength
            iv["add"][s] = add * sign * unit * e[s]
    for s in patch_sites:
        if patch == "full":
            iv["patch"][s] = np.eye(d_model)
        elif patch in ("pca", "pca_all"):
            u = fitted["basis" if patch == "pca" else "basis_all"][s, rock, :rank]
            iv["patch"][s] = u.T @ u
        elif patch == "random":
            u = np.linalg.qr(np.random.default_rng(null_seed + 104729 * rock + s).normal(
                size=(d_model, rank)))[0].T
            iv["patch"][s] = u.T @ u
        elif patch == "encoder":
            u = fitted["encoder"][encoder][s, rock]
            u = u / np.linalg.norm(u)
            iv["patch"][s] = np.outer(u, u)
    rng = np.random.default_rng(null_seed + 7919 * rock)
    for s in range(SITES):
        q = np.linalg.qr(w[s])[0]
        z = rng.normal(size=d_model)
        z -= q @ (q.T @ z)
        iv["null"][s] = z / np.linalg.norm(z)
    return {k: jnp.asarray(v, dtype=bool if k in ("control", "use_proto") else jnp.float32)
            for k, v in iv.items()}


@partial(jax.jit, static_argnames=("spec", "env", "episodes"))
def rollout(spec, env, params, iv, *, episodes, seed, rock, good, random_policy=False):
    """Closed-loop episodes with one rock's quality fixed and an optional intervention.

    A never-intervened twin model sees the same history with the target rock's
    sensor symbols flipped; ``iv['patch']`` moves activations towards the twin.
    """
    obs, state = jax.vmap(partial(rs.reset, env))(
        jax.random.split(jax.random.key(seed), episodes))
    state = state._replace(qualities=state.qualities.at[:, rock].set(jnp.asarray(good, bool)))
    initial_good = state.qualities
    cache = jax.vmap(lambda _: tm.empty_cache(spec))(jnp.arange(episodes))
    column = check_index(env, rock)
    rocks = jnp.asarray(env.rocks)

    def encode(c, tc, o, active):
        is_check = o[column] > .5
        twin_sites = []
        def capture(_, x):
            twin_sites.append(x.reshape(-1))
            return x
        _, tc = tm.encode_step(spec, params, tc, flip_symbols(o, active & is_check, env),
                               residual_transform=capture)
        tw = jnp.stack(twin_sites)
        gp = jnp.where(is_check, iv["gate_patch"][0], iv["gate_patch"][1]) * active
        gs = jnp.where(is_check, iv["gate_steer"][0], iv["gate_steer"][1]) * active
        stats = []
        def transform(i, x):
            h = x.reshape(-1)
            before = h @ iv["d"][i] + iv["b"][i]
            delta = (gs * (iv["lam"][i] * (iv["target"] - before) * iv["e"][i] + iv["add"][i]
                           + h @ iv["lin"][i] + iv["off"][i])
                     + gp * (iv["patch"][i] @ (jnp.where(iv["use_proto"], iv["proto"][i], tw[i]) - h)))
            delta = jnp.where(iv["control"], jnp.linalg.norm(delta) * iv["null"][i], delta)
            after = (h + delta) @ iv["d"][i] + iv["b"][i]
            stats.append(jnp.stack((before, after, jnp.linalg.norm(delta))))
            readout.append((h + delta) @ iv["ro_w"] + iv["ro_b"])
            return (h + delta).reshape(x.shape)
        readout = []
        z, c = tm.encode_step(spec, params, c, o, residual_transform=transform)
        return z, c, tc, jnp.stack(stats), readout[-1]

    def advance(carry, _):
        obs, state, cache, tcache, done, sampled, key = carry
        active = ~sampled[:, rock]
        z, cache, tcache, stats, readout = jax.vmap(encode)(cache, tcache, obs, active)
        key, ak, ek = jax.random.split(key, 3)
        logits, _ = tm.heads(params, z)
        logits = jnp.where(random_policy, jnp.zeros_like(logits), logits)
        action = jax.random.categorical(ak, logits).astype(jnp.int32)
        out = jax.vmap(partial(rs.step, env))(jax.random.split(ek, episodes), state, action)
        valid = ~done
        picked = ((action == 4)[:, None] & jnp.all(state.rover[:, None] == rocks, axis=-1)
                  & valid[:, None])
        live = valid & active
        unsampled = valid[:, None] & ~sampled
        row = (readout * unsampled, unsampled, picked, (action == 5 + rock) & live, out.reward * valid,
               stats * live[:, None, None], live,
               jnp.all(state.rover == rocks[rock], axis=-1) & live, out.terminated & valid, valid)
        return (out.obs, out.state, cache, tcache, done | out.terminated | out.truncated,
                sampled | picked, key), row

    carry = (obs, state, cache, cache, jnp.zeros(episodes, bool),
             jnp.zeros((episodes, env.k), bool), jax.random.key(seed + 1))
    _, rows = jax.lax.scan(advance, carry, jnp.arange(env.episode_length))
    readout, unsampled, picked, checks, rewards, stats, live, visits, exits, valid = rows
    return dict(readout=readout.sum(0), readout_steps=unsampled.sum(0),
                readout_time=readout[:, :, rock].sum(1), readout_time_steps=unsampled[:, :, rock].sum(1),
                picked=picked.any(0), checks=checks.sum(0), returns=rewards.sum(0),
                stats=stats.sum((0, 1)), live=live.sum(), initial_good=initial_good,
                visits=visits.any(0), exits=exits.any(0), length=valid.sum(0))


def wilson(successes, n):
    p = successes / n
    center = (p + Z95**2 / (2 * n)) / (1 + Z95**2 / n)
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / (1 + Z95**2 / n)
    return [float(center - half), float(center + half)]


def summarize(values, rock):
    v = {k: np.asarray(x) for k, x in values.items()}
    n = len(v["returns"])
    other = np.arange(v["picked"].shape[1]) != rock
    good = v["initial_good"][:, other]
    picked = v["picked"][:, other]
    live = max(int(v["live"]), 1)
    target = int(v["picked"][:, rock].sum())
    read = lambda mask: float((v["readout"] * mask).sum() / max((v["readout_steps"] * mask).sum(), 1))
    full = np.ones_like(v["initial_good"])
    full[:, rock] = False
    return dict(episodes=n, target=target / n, target_ci95=wilson(target, n),
                other_good=float((picked & good).sum() / good.sum()),
                other_bad=float((picked & ~good).sum() / (~good).sum()),
                other_good_counts=[int((picked & good).sum()), int(good.sum())],
                other_bad_counts=[int((picked & ~good).sum()), int((~good).sum())],
                mean_return=float(v["returns"].mean()), checks=float(v["checks"].mean()),
                visit=float(v["visits"].mean()), exit=float(v["exits"].mean()),
                length=float(v["length"].mean()),
                readout_target=read(~full), readout_other_good=read(full & v["initial_good"]),
                readout_other_bad=read(full & ~v["initial_good"]),
                readout_time=(v["readout_time"] / np.maximum(v["readout_time_steps"], 1)).tolist(),
                readout_time_steps=v["readout_time_steps"].tolist(),
                decoded_before=(v["stats"][:, 0] / live).tolist(),
                decoded_after=(v["stats"][:, 1] / live).tolist(),
                shift_norm=(v["stats"][:, 2] / live).tolist())
