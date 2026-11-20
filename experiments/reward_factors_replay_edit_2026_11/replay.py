"""Action-aligned replay and residual-displacement analysis for independent factors."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
from experiments.belief_quotient_history_jax_2026_11 import env
from experiments.belief_quotient_replay_edit_2026_11 import replay as ladder

LAGS = ladder.LAGS
WARMUP = base.WARMUP
RCOND = 1e-6
ACTION = base.ACTION_TABLE
INVERSE = np.empty((3, 3), np.int32)
INVERSE[tuple(ACTION.T)] = np.arange(9)
TOKEN_SWAP = np.array([1, 0, 2])
ACTION_SWAP = np.array([0, 2, 1])


def action_permutation(frame):
    a = ACTION.copy()
    for f in range(2):
        if frame & (1 << f):
            a[:, f] = ACTION_SWAP[a[:, f]]
    return INVERSE[tuple(a.T)]


PERM = np.stack([action_permutation(i) for i in range(4)])


def mirror_inputs(obs, frame, xp=np):
    indices = xp.arange(9)
    lead, follow = indices // 3, indices % 3
    swap = xp.asarray(TOKEN_SWAP)
    perm = xp.where(frame & 1, swap[lead], lead) * 3 + xp.where(frame & 2, swap[follow], follow)
    return xp.concatenate((obs[..., :9][..., perm], obs[..., 9:][..., xp.asarray(PERM)[frame]]), -1)


def fold_actions(prob):
    joint = prob[..., INVERSE]
    return np.stack((joint.sum(-1), joint.sum(-2)), axis=-2)


def factors(environment, obs):
    token = obs[..., :9].argmax(-1)
    tokens = np.stack(np.divmod(token, 3), -1)
    actions = ACTION[obs[..., 9:].argmax(-1)]
    e = environment.emission_matrix()
    tr = environment.action_transitions(3)
    b = np.broadcast_to(environment.INITIAL_STATE_DISTRIBUTION, (len(obs), 2, 3))
    out = np.empty(tokens.shape + (3,))
    for i in range(tokens.shape[1]):
        if i:
            b = np.einsum("bfs,bfsj->bfj", b, tr[actions[:, i]])
        b = b * np.moveaxis(e[:, tokens[:, i]], 0, -1)
        b = b / b.sum(-1, keepdims=True)
        out[:, i] = b
    return out


def edit(obs, factor, pos, rng):
    token = obs[..., :9].argmax(-1)
    digits = list(np.divmod(token, 3))
    val = digits[factor][:, pos]
    digits[factor][:, pos] = np.where(val == 2, rng.integers(2, size=val.shape), 2)
    result = obs.copy()
    result[..., :9] = np.eye(9, dtype=obs.dtype)[3 * digits[0] + digits[1]]
    return result


@jax.jit
def observations(process, actions, key):
    reset_key, step_key, _ = jax.random.split(key, 3)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, len(actions)))
    def step(state, item):
        t, action = item
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        state, _, _ = jax.vmap(env.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), len(actions)), state, action)
        return state, obs
    return jax.lax.scan(step, state, (jnp.arange(actions.shape[1]), actions.T))[1].swapaxes(0, 1)


def collect(spec, params, process, seed, episodes, length):
    key = base._keys(seed)[0]
    record = base.collect(spec, params, process, episodes, length, key)
    return dict(jax.device_get(record), obs=np.asarray(observations(process, record["action"], key)))


def profile(z, zz, h, hh, drop, aligned):
    p, q = [fold_actions(np.exp(ladder.log_probs(x))) for x in (z, zz)]
    tv = ladder.tv(p, q)
    rel = np.linalg.norm(hh - h, axis=-1) / np.maximum(np.linalg.norm(h, axis=-1), 1e-12)
    high = drop >= ladder.DROP
    return dict(tv=tv.mean(0).tolist(), relative=rel.mean(0).tolist(),
                  drop=float(drop.mean()), high_drop_n=int(high.sum()), n=int(high.size),
                  high_drop_tv=tv[high].mean(0).tolist() if high.any() else None,
                  high_drop_relative=rel[high].mean(0).tolist() if high.any() else None,
                  aligned_tv=ladder.tv(p, fold_actions(np.exp(ladder.log_probs(aligned)))).mean(0).tolist())


def marginal_fidelity(z, z_ref, z_written, drop):
    p, q, r = [fold_actions(np.exp(ladder.log_probs(x))) for x in (z, z_ref, z_written)]
    effect, error = ladder.tv(p, q), ladder.tv(r, q)
    result = {}
    for label, mask in (("all", np.ones_like(drop, bool)), ("low", drop < ladder.DROP), ("high", drop >= ladder.DROP)):
        n = int(mask.sum())
        total = effect[mask].sum(0)
        result[label] = dict(n=n, replay_tv=(total / n).tolist() if n else [None, None],
                             fidelity=[float(1 - e / a) if n and a >= 1e-6 * n else None
                                       for e, a in zip(error[mask].sum(0), total)],
                             mean_drop=float(drop[mask].mean()) if n else None)
    return result


def subspaces(h, mirrors, rank=8):
    scores = []
    for k in range(h.shape[-2]):
        basis, coverage = [], []
        for hh in mirrors:
            v = (hh[:, WARMUP:, k] - h[:, WARMUP:, k]).reshape(-1, h.shape[-1])
            v -= v.mean(0)
            e, u = np.linalg.eigh(v.T @ v)
            basis.append(u[:, -rank:])
            coverage.append(float(e[-rank:].sum() / max(e.sum(), 1e-12)))
        singular = np.linalg.svd(basis[0].T @ basis[1], compute_uv=False)
        scores.append(dict(overlap=float(np.square(singular).mean()), coverage=coverage))
    return scores


def replay_cases(spec, params, environment, records, seed, patches=False, factor_ids=(0, 1),
                 max_episodes=12, part_e=False):
    obs, actions = (records[k][:max_episodes] for k in ("obs", "action"))
    z, h = map(np.asarray, ladder.forward(spec, params, obs))
    truth = factors(environment, obs)
    joint = records["belief"][:max_episodes]
    expected = (truth[..., 0, :, None] * truth[..., 1, None, :]).reshape(joint.shape)
    np.testing.assert_allclose(joint, expected, atol=2e-5)
    end = np.arange(255, obs.shape[1], 256)
    rng = np.random.default_rng(seed)
    a, b, mirrors, data = {}, {}, [], {}
    for f in factor_ids:
        pairs = []
        cases = [(lag, two) for lag in LAGS for two in (False, True) if not two or lag in (8, 32)]
        if not part_e:
            cases += [(None, False)]
        for lag, two in cases:
            if lag is None:
                obs_alt = mirror_inputs(obs, 1 << f)
                query = np.arange(WARMUP, obs.shape[1], 4)
                name = f"f{f + 1}_mirror"
            else:
                pos = np.union1d(end - lag, end) if two else end - lag
                obs_alt = edit(obs, f, pos, rng)
                query = np.unique(np.concatenate([np.arange(t - lag, t + 1, max(1, (lag + 1) // 8)) for t in end] + [end]))
                name = f"f{f + 1}_{lag}_{int(two)}"
            changed = np.any(obs_alt != obs, -1)
            known = factors(environment, obs_alt)
            np.testing.assert_array_equal(known[..., 1 - f, :], truth[..., 1 - f, :])
            zz, hh = map(np.asarray, ladder.forward(spec, params, obs_alt))
            if lag is None:
                np.testing.assert_allclose(known[..., f, :], truth[..., f, TOKEN_SWAP], atol=1e-10)
                mirrors.append(hh)
                aligned = zz[..., PERM[1 << f]]
                drop = ladder.action_drop(z, aligned, actions)
                averaged = np.broadcast_to(drop[:, WARMUP:].mean(1, keepdims=True), drop.shape)
                a[name] = profile(*(x[:, WARMUP:] for x in (z, zz, h, hh, averaged, aligned)))
                a[name]["first_position"] = WARMUP
                evaluate = slice(WARMUP, None, 8)
            else:
                aligned = None
                drop = ladder.action_drop(z, zz, actions)
                averaged = np.zeros_like(drop)
                start = query // 256 * 256 + 255 - lag
                averaged[:, query] = np.stack([drop[:, i:j + 1].mean(1) for i, j in zip(start, query)], 1)
                evaluate = end
                # Event-aligned profiles cover every position from the first edit to the query.
                t = np.stack([np.arange(q - lag, q + 1) for q in end])
                high = averaged[:, end] >= ladder.DROP
                p, pp = [fold_actions(np.exp(ladder.log_probs(x[:, t]))) for x in (z, zz)]
                rel = np.linalg.norm((hh - h)[:, t], axis=-1) / np.maximum(np.linalg.norm(h[:, t], axis=-1), 1e-12)
                diff = ladder.tv(p, pp)
                pb = np.take_along_axis(ladder.log_probs(z), actions[..., None], -1)[..., 0]
                pr = np.take_along_axis(ladder.log_probs(zz), actions[..., None], -1)[..., 0]
                probability_drop = (np.exp(pb) - np.exp(pr))[:, t]
                belief_tv = ladder.tv(truth[:, t, f, :], known[:, t, f, :])
                a[name] = dict(tv=diff.mean((0, 1)).tolist(), relative=rel.mean((0, 1)).tolist(),
                               high_drop_n=int(high.sum()), n=high.size, drop=float(averaged[:, end].mean()),
                               probability_drop=probability_drop.mean((0, 1)).tolist(),
                               belief_tv=belief_tv.mean((0, 1)).tolist(), unchanged_factor_max_error=0.,
                               high_drop_probability_drop=probability_drop[high].mean(0).tolist() if high.any() else None,
                               high_drop_tv=diff[high].mean(0).tolist() if high.any() else None,
                               high_drop_relative=rel[high].mean(0).tolist() if high.any() else None)
            if not part_e:
                tokens = one_factor_token(obs, f)
                tok_alt = one_factor_token(obs_alt, f)
                act_delta = obs_alt[..., 9:] - obs[..., 9:]
                nuisance = np.concatenate((tok_alt - tokens, act_delta), -1)
                pairs.append(tuple(x[:, query].reshape(-1, *x.shape[2:]) for x in
                                   (truth, known, h, hh, nuisance, averaged)))
            if patches:
                latest = np.zeros_like(changed)
                latest[:, end] = True
                cells = []
                for site in range(5):
                    masks = dict(latest=latest, changed=changed, past=changed & ~latest)
                    if lag is None:
                        # Each queried decision gets a separate past-without-latest intervention.
                        # Earlier queried positions are excluded only in their own counterfactual.
                        one = np.arange(WARMUP, obs.shape[1], 256)
                        q_orig, q_alt = z[:, one], zz[:, one]
                        drop_q = averaged[:, one]
                        cells.append(mirror_patch(spec, params, h[..., site, :], hh[..., site, :],
                                                  changed, one, site, q_orig, q_alt, drop_q))
                        continue
                    cells.append({mode: marginal_fidelity(z[:, evaluate], zz[:, evaluate],
                                  np.asarray(ladder.patch(spec, params, h[..., site, :], hh[..., site, :],
                                                          mask, site))[:, evaluate],
                                  averaged[:, evaluate]) for mode, mask in masks.items()})
                b[name] = cells
        if pairs:
            data[f] = tuple(np.concatenate(x) for x in zip(*pairs))
    if len(mirrors) == 2:
        a["subspaces"] = subspaces(h, mirrors)
    return a, b, data


def mirror_patch(spec, params, h, hh, changed, query, site, z, zz, drop):
    result = {}
    for name in ("latest", "changed", "past"):
        pred = []
        for t in query:
            mask = np.zeros_like(changed) if name == "latest" else changed.copy()
            if name != "changed":
                mask[:, t] = name == "latest"
            pred.append(np.asarray(ladder.patch(spec, params, h, hh, mask, site))[:, t])
        result[name] = marginal_fidelity(z, zz, np.stack(pred, 1), drop)
    return result


def one_factor_token(obs, factor):
    token = obs[..., :9].reshape(obs.shape[:-1] + (3, 3))
    return token.sum(-1 if factor == 0 else -2)


FAMILIES = tuple((log, quad, condition) for log in (False, True) for quad in (False, True)
                 for condition in ("none", "other", "pc", "both"))


def features(b, log, quad, xp):
    if log:
        v = xp.log(xp.maximum(b, 1e-4))
        v = v[..., :2] - v[..., 2:]
    else:
        v = b[..., :2]
    if quad:
        v = xp.concatenate((v, v[..., :1] ** 2, v[..., :1] * v[..., 1:], v[..., 1:] ** 2), -1)
    return v


def design(b, bt, h, nuisance, f, family, center, basis, xp=np):
    log, quad, cond = family
    delta = features(bt[..., f, :], log, quad, xp) - features(b[..., f, :], log, quad, xp)
    parts = [delta]
    if cond in ("other", "both"):
        other = b[..., 1 - f, :2] - 1 / 3
        parts.append((delta[..., :, None] * other[..., None, :]).reshape(delta.shape[:-1] + (-1,)))
    if cond in ("pc", "both"):
        pc = (h - center) @ basis
        parts.append((delta[..., :, None] * pc[..., None, :]).reshape(delta.shape[:-1] + (-1,)))
    return xp.concatenate((*parts, nuisance), -1)


def fit(train, validation, test):
    result, maps = {}, {}
    for f in train:
        result[f], maps[f] = [], []
        for site in range(5):
            def rows(d):
                b, bt, h, ht, nuisance, drop = d[f]
                h, ht = h[:, site].astype(float), ht[:, site].astype(float)
                return (b, bt, h, nuisance), ht - h, drop
            x, y, _ = rows(train)
            val, vy, _ = rows(validation)
            held, hy, drop = rows(test)
            center = x[2].mean(0)
            _, u = np.linalg.eigh((x[2] - center).T @ (x[2] - center))
            pc = u[:, -4:]
            scores, winner = {}, None
            for family in FAMILIES:
                label = "_".join(map(str, family))
                coef = np.linalg.lstsq(design(*x, f, family, center, pc), y, rcond=RCOND)[0]
                vr = ladder.r2(vy, design(*val, f, family, center, pc) @ coef)
                pred = design(*held, f, family, center, pc) @ coef
                fixed = ~held[-1].any(-1)
                scores[label] = dict(validation=vr, heldout=ladder.r2(hy, pred),
                                     fixed_inputs=ladder.r2(hy[fixed], pred[fixed]),
                                     high_drop=ladder.r2(hy[drop >= ladder.DROP], pred[drop >= ladder.DROP]),
                                     low_drop=ladder.r2(hy[drop < ladder.DROP], pred[drop < ladder.DROP]))
                if winner is None or vr > winner[0]:
                    winner = vr, label, family, coef
            _, label, family, coef = winner
            maps[f].append((family, tuple(jnp.asarray(a, jnp.float32) for a in (center, pc, coef))))
            result[f].append(dict(winner=label, models=scores, n=len(drop), high_drop_n=int((drop >= ladder.DROP).sum())))
    return result, maps
