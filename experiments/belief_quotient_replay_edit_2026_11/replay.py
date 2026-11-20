"""Action-fixed filters, interventions and paired displacement fits."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import log_softmax

from experiments.algorithm_generality_probes_eps02_2026_11 import paper_coarse, paper_probe, steer_core as old
from experiments.belief_quotient_history_jax_2026_11 import models

LAGS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
WARMUP = 8
DROP = 0.5
SWAP_TOKEN = np.array([1, 0, 2])
SWAP_ACTION = np.array([0, 2, 1])
FEATURE_SETS = ("c", "rho", "rho_delta", "rho_abs_delta")
FAMILIES = tuple((key, log, quadratic, pc) for key in FEATURE_SETS
                 for log in (False, True) for quadratic in (False, True) for pc in (False, True))


def filtered(process, obs, variant):
    t = obs[..., :3].argmax(-1)
    a = obs[..., 3:].argmax(-1)
    pred = np.broadcast_to(process.initial, (len(t), 3)).astype(float)
    b = np.empty(t.shape + (3,))
    tr, e = np.asarray(process.transitions, float), np.asarray(process.emission, float)
    for i in range(t.shape[1]):
        if i:
            pred = np.einsum("bi,bij->bj", b[:, i - 1], tr[a[:, i]])
        z = pred * e[:, t[:, i]].T
        b[:, i] = z / z.sum(-1, keepdims=True)
    if variant == 2:
        c = paper_probe.coarse_filter(obs, process._asdict())
    else:
        c = paper_coarse.merged_filter(obs, process._asdict())[..., 2]
    return np.concatenate([b, c[..., None]], -1)


def edit(obs, name, rng, full=False, positions=None):
    replay = obs.copy()
    t = obs[..., :3].argmax(-1)
    mask = np.zeros(t.shape, bool)
    if name == "swap":
        u = SWAP_TOKEN[t]
        if full:
            replay[..., 3:] = replay[..., 3:][..., SWAP_ACTION]
    elif name == "relabel":
        u = np.where(t == 2, 2, rng.integers(2, size=t.shape))
    else:
        if positions is None:
            mask[np.arange(len(t)), rng.integers(WARMUP, t.shape[1], size=len(t))] = True
        else:
            mask[:, positions] = True
        u = np.where(mask, np.where(t == 2, rng.integers(2, size=t.shape), 2), t)
    replay[..., :3] = np.eye(3, dtype=obs.dtype)[u]
    return replay, u != t


def block(spec, p, x):
    q, k, v = jnp.split(models._dense(p["qkv"], models._layer_norm(p["ln1"], x)), 3, -1)
    shape = x.shape[:2] + (spec.n_heads, spec.head_dim)
    pos = jnp.arange(x.shape[1])[:, None]
    q, k = (models._rope(y.reshape(shape), pos) for y in (q, k))
    attn = models._banded_attention(q, k, v.reshape(shape), spec.context_len)
    x = x + models._dense(p["proj"], attn)
    return x + models._mlp(p["mlp"], models._layer_norm(p["ln2"], x))


@partial(jax.jit, static_argnums=(0,))
def forward(spec, params, obs):
    x = models._dense(params["input"], obs)
    sites = []
    for p in params["blocks"]:
        x = block(spec, p, x)
        sites.append(x)
    sites.append(models._layer_norm(params["final_norm"], x))
    return models._dense(params["policy"], sites[-1]), jnp.stack(sites, -2)


@partial(jax.jit, static_argnums=(0, 5))
def patch(spec, params, orig, ref, mask, site):
    x = jnp.where(mask[..., None], ref, orig)
    for p in params["blocks"][site + 1:]:
        x = block(spec, p, x)
    if site < 4:
        x = models._layer_norm(params["final_norm"], x)
    return models._dense(params["policy"], x)


def tv(p, q):
    return np.abs(p - q).sum(-1) * .5


def log_probs(logits):
    return log_softmax(logits / 1.5, axis=-1)


def action_drop(base, changed, action):
    indices = action[..., None]
    return np.take_along_axis(log_probs(base) - log_probs(changed), indices, -1)[..., 0]


def fidelity(base, ref, written, drop):
    original, counterfactual, replacement = (np.exp(log_probs(z)) for z in (base, ref, written))
    effect = tv(original, counterfactual)
    error = tv(replacement, counterfactual)
    answer = {}
    for name, mask in (("all", np.ones_like(drop, bool)), ("low_drop", drop < DROP), ("high_drop", drop >= DROP)):
        n = int(mask.sum())
        denom = float(effect[mask].sum())
        answer[name] = dict(n=n, fidelity=float(1 - error[mask].sum() / denom) if denom > 1e-9 else None,
                            replay_tv=float(effect[mask].mean()) if n else None,
                            mean_drop=float(drop[mask].mean()) if n else None)
    return answer


def r2(actual, pred):
    if not len(actual):
        return None
    den = np.square(actual - actual.mean(0)).sum()
    return float(1 - np.square(actual - pred).sum() / den) if den > 0 else None


def invariance(spec, params, process, variant, train, test):
    logits, h = map(np.asarray, forward(spec, params, test["obs"]))
    true = filtered(process, test["obs"], variant)
    out = {}
    for number, name in enumerate(("swap", "relabel", "flip")):
        alt, _ = edit(test["obs"], name, np.random.default_rng(8100 + number), variant == 3)
        counter = filtered(process, alt, variant)
        if name == "swap":
            np.testing.assert_allclose(true[..., :3], counter[..., SWAP_TOKEN], atol=1e-10)
        if variant == 2 and name != "flip":
            np.testing.assert_allclose(true[..., 3], counter[..., 3], atol=1e-10)
        z, twin = map(np.asarray, forward(spec, params, alt))
        p, q = np.exp(log_probs(logits)), np.exp(log_probs(z))
        drop = action_drop(logits, z, test["action"])
        relative = np.linalg.norm(twin - h, axis=-1) / np.maximum(np.linalg.norm(h, axis=-1), 1e-12)
        mask = np.broadcast_to(drop[:, WARMUP:].mean(1, keepdims=True) >= DROP, drop.shape)
        out[name] = dict(
            tv=tv(p, q).mean(0).tolist(), greedy=(p.argmax(-1) != q.argmax(-1)).mean(0).tolist(),
            relative=relative.mean(0).tolist(), drop=drop.mean(0).tolist(),
            belief_mae=np.abs(true - counter)[:, WARMUP:].mean((0, 1)).tolist(),
            high_drop_windows=int(mask[:, 0].sum()),
            high_drop_tv=float(tv(p, q)[mask].mean()) if mask.any() else None,
            high_drop_greedy=float((p.argmax(-1) != q.argmax(-1))[mask].mean()) if mask.any() else None,
            high_drop_relative=relative[mask].mean(0).tolist() if mask.any() else None,
        )
        if variant == 3 and name == "swap":
            out[name]["symmetry_tv"] = tv(p, q[..., SWAP_ACTION]).mean(0).tolist()
        if name == "relabel":
            obs = train["obs"]
            alt_fit, _ = edit(obs, name, np.random.default_rng(7201))
            dx = (filtered(process, alt_fit, variant) - filtered(process, obs, variant))[:, WARMUP:, 2].reshape(-1, 1)
            dz = (np.asarray(forward(spec, params, alt_fit)[0]) - train["logits"])[:, WARMUP:].reshape(-1, 3)
            coef, bias = old.affine(dx, dz)
            y = (z - logits)[:, WARMUP:].reshape(-1, 3)
            x = (counter - true)[:, WARMUP:, 2].reshape(-1, 1)
            out[name].update(rho_logit_r2=r2(y, x @ coef + bias), rho_logit_slopes=coef[0].tolist())
    fit = filtered(process, train["obs"], variant)
    def targets(b):
        delta = b[..., 0] - b[..., 1]
        return np.stack((b[..., 2], delta, abs(delta), b[..., 3]), -1)[:, WARMUP:].reshape(-1, 4)
    out["probe"] = []
    for site in range(5):
        x = train["h"][:, WARMUP:, site].reshape(-1, h.shape[-1])
        v = h[:, WARMUP:, site].reshape(-1, h.shape[-1])
        w, bias = old.affine(x, targets(fit))
        out["probe"].append([r2(y, estimate) for y, estimate in zip(targets(true).T, (v @ w + bias).T)])
    return out


def collect_pairs(spec, params, process, variant, original, seed, evaluate_patches):
    obs, z, h = (original[k] for k in ("obs", "logits", "h"))
    b = filtered(process, obs, variant)
    rng = np.random.default_rng(seed)
    end = np.arange(255, obs.shape[1], 256)
    # Successive queries' attention bands do not overlap.
    patches, pairs = {}, []
    cases = [(lag, both) for lag in LAGS for both in (False, True) if lag or not both]
    for lag, both in cases + [(None, False)]:
        if lag is None:
            alt, mask = edit(obs, "relabel", rng)
            query = np.arange(WARMUP, obs.shape[1], 4)
        else:
            pos = np.union1d(end - lag, end) if both else end - lag
            alt, mask = edit(obs, "flip", rng, positions=pos)
            query = np.unique(np.concatenate([np.arange(i - lag, i + 1, max(1, (lag + 1) // 8)) for i in end] + [end]))
        z_alt, h_alt = map(np.asarray, forward(spec, params, alt))
        b_alt = filtered(process, alt, variant)
        drop = action_drop(z, z_alt, original["action"])
        if lag is None:
            pair_drop = np.broadcast_to(drop[:, WARMUP:].mean(1, keepdims=True), drop.shape)
        else:
            start = (query // 256) * 256 + 255 - lag
            pair_drop = np.zeros_like(drop)
            pair_drop[:, query] = np.stack([drop[:, i:j + 1].mean(1) for i, j in zip(start, query)], 1)
        pairs.append(tuple(x[:, query].reshape(-1, *x.shape[2:]) for x in
                           (b, b_alt, h, h_alt, obs[..., :3], alt[..., :3], pair_drop)))
        if lag is None:
            continue
        drop = pair_drop[:, end]
        if evaluate_patches:
            latest = np.zeros_like(mask)
            latest[:, end] = True
            modes = dict(latest=latest, changed=mask, past=mask & ~latest)
            patches[f"{lag}_{int(both)}"] = [
                {mode: fidelity(z[:, end], z_alt[:, end],
                                np.asarray(patch(spec, params, h[..., site, :], h_alt[..., site, :], where, site))[:, end],
                                drop) for mode, where in modes.items()}
                for site in range(5)]
    return patches, tuple(np.concatenate(columns) for columns in zip(*pairs))


def features(b, key, log, quadratic, xp=np):
    rho = b[..., 3] if key == "c" else b[..., 2]
    delta = b[..., 0] - b[..., 1]
    if log:
        prob = xp.clip(rho, 1e-4, 1 - 1e-4)
        rho = xp.log(prob) - xp.log1p(-prob)
        delta = xp.log(xp.maximum(b[..., 0], 1e-4)) - xp.log(xp.maximum(b[..., 1], 1e-4))
    parts = [rho]
    if key in ("rho_delta", "rho_abs_delta"):
        parts.append(delta if key == "rho_delta" else xp.abs(delta))
    if quadratic:
        parts += [x * y for i, x in enumerate(parts) for y in parts[i:]]
    return xp.stack(parts, -1)


def design(b, target, h, token_delta, family, mean, pc, xp=np):
    key, log, quadratic, interaction = family
    f = features(target, key, log, quadratic, xp) - features(b, key, log, quadratic, xp)
    if interaction:
        v = (h - mean) @ pc
        f = xp.concatenate((f, (f[..., :, None] * v[..., None, :]).reshape(f.shape[:-1] + (-1,))), -1)
    return xp.concatenate((f, token_delta[..., :2]), -1)


def fit_encoders(train, select, test):
    fitted, scores = [], []
    for site in range(5):
        def arrays(data):
            b, target, h, twin, token, alt_token, _ = data
            h, twin = h[:, site].astype(float), twin[:, site].astype(float)
            return (b, target, h, alt_token - token), twin - h
        (b, bt, h, token), dy = arrays(train)
        sel, sy = arrays(select)
        held, hy = arrays(test)
        drop = test[-1]
        mean = h.mean(0)
        _, vecs = np.linalg.eigh((h - mean).T @ (h - mean))
        pc = vecs[:, -4:]
        results, best = {}, None
        for family in FAMILIES:
            # No offset: an unchanged belief and token must produce zero write.
            coef = np.linalg.lstsq(design(b, bt, h, token, family, mean, pc), dy, rcond=old.AFFINE_RCOND)[0]
            sr = r2(sy, design(*sel, family, mean, pc) @ coef)
            hx = design(*held, family, mean, pc) @ coef
            unchanged = ~held[-1].any(-1)
            label = "_".join(map(str, family))
            results[label] = dict(selection_r2=sr, heldout_r2=r2(hy, hx),
                                  heldout_fixed_token_r2=r2(hy[unchanged], hx[unchanged]),
                                  heldout_low_drop_r2=r2(hy[drop < DROP], hx[drop < DROP]),
                                  heldout_high_drop_r2=r2(hy[drop >= DROP], hx[drop >= DROP]))
            if best is None or sr > best[0]:
                best = sr, label, family, coef
        _, label, family, coef = best
        fitted.append((family, mean.astype("float32"), pc.astype("float32"), coef.astype("float32")))
        scores.append(dict(winner=label, models=results, high_drop_rows=int((drop >= DROP).sum()),
                           total_rows=len(drop), mean_drop=float(drop.mean())))
    return fitted, scores


def records(spec, process, params, seed, episodes=12, length=1024):
    raw = old.rollout(spec, process, params, old.empty_maps(), jax.random.PRNGKey(seed),
                      episodes, length, capture=True)
    return {k: np.asarray(v).swapaxes(0, 1) for k, v in raw.items()}
