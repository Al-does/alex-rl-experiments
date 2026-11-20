"""Post-fit comparisons with truth; never imported by the distillation core."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core
from .data import quotient_model


def flat(x):
    return np.asarray(x).reshape(int(np.prod(x.shape[:-1])), x.shape[-1])


def affine(x, y):
    with jax.default_matmul_precision("highest"):
        X = jnp.concatenate((jnp.asarray(x), jnp.ones((len(x), 1))), -1)
        return np.asarray(jnp.linalg.lstsq(X, jnp.asarray(y), rcond=1e-6)[0])


def predict(x, w):
    return np.concatenate((x, np.ones((len(x), 1))), -1) @ w


def r2(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if not y.size or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = np.square(y-y.mean(0)).mean()
        scale = max(1., float(np.square(y).mean()))
        if not np.isfinite(denominator) or denominator <= 1e-12*scale:
            return None
        value = float(1-np.square(y-prediction).mean()/denominator)
    return value if np.isfinite(value) else None


def finite_metrics(value):
    reasons = {}
    def clean(item, path):
        if isinstance(item, dict):
            return {key: clean(v, f"{path}.{key}" if path else key) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v, f"{path}[{i}]") for i, v in enumerate(item)]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            reasons[path] = "non-finite metric"
            return None
        return item
    result = clean(value, "")
    if reasons and isinstance(result, dict):
        result["nonfinite_metrics"] = reasons
    return result


def mlp_score(x, y, test_x, test_y, *, updates=300):
    if not all(np.isfinite(a).all() for a in (x, y, test_x, test_y)):
        return None
    if not x.size or np.all(np.std(x, axis=0) <= 1e-12):
        return r2(test_y, np.broadcast_to(y.mean(0), test_y.shape))
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-4)
    x, test_x = (x-mean)/scale, (test_x-mean)/scale
    key1, key2 = jax.random.split(jax.random.PRNGKey(515))
    p = (jax.random.normal(key1, (x.shape[-1], 32))*.1, jnp.zeros(32),
         jax.random.normal(key2, (32, y.shape[-1]))*.1, jnp.zeros(y.shape[-1]))
    def output(p, x):
        return jnp.tanh(x @ p[0] + p[1]) @ p[2] + p[3]
    optimizer = optax.adam(.003)
    opt = optimizer.init(p)
    @jax.jit
    def step(p, opt, x, y):
        grad = jax.grad(lambda p: jnp.square(output(p, x)-y).mean())(p)
        delta, opt = optimizer.update(grad, opt, p)
        return optax.apply_updates(p, delta), opt
    rng = np.random.default_rng(515)
    for _ in range(updates):
        ix = rng.integers(len(x), size=min(1024, len(x)))
        p, opt = step(p, opt, x[ix], y[ix])
    return r2(test_y, np.asarray(output(p, test_x)))


def geometry(train_g, test_g, train_truth, test_truth, *, mlp_updates=300):
    x, tx = flat(train_g), flat(test_g)
    report = {}
    for name, target in train_truth.items():
        y, ty = flat(target), flat(test_truth[name])
        w, reverse = affine(x, y), affine(y, x)
        report[name] = {"affine_r2": r2(ty, predict(tx, w)),
                        "mlp_r2": mlp_score(x, y, tx, ty, updates=mlp_updates),
                        "reverse_affine_r2": r2(tx, predict(ty, reverse)),
                        "reverse_mlp_r2": mlp_score(y, x, ty, tx, updates=mlp_updates) if x.shape[-1] else None}
        for metric, value in list(report[name].items()):
            if value is None:
                report[name][metric+"_reason"] = "empty, non-finite, or near-constant target/prediction"
    return report


def commutation(spec, p, train_g, test_g, train_b, test_b, test_x, process):
    """Unprojected affine predictions; invalid-simplex frequency is explicit."""
    w = affine(flat(train_g), flat(train_b))
    before = predict(flat(test_g[:, :-1]), w)
    x = test_x[:, 1:].reshape(-1)
    updated = core.update(spec, p, jnp.asarray(flat(test_g[:, :-1])), jnp.asarray(x))
    after = predict(np.asarray(updated), w)
    T, E = np.asarray(process.transitions), np.asarray(process.emission)
    def bayes(b):
        b = np.einsum("bi,bij->bj", b, T[x//spec.tokens]) * E[:, x % spec.tokens].T
        total = b.sum(-1, keepdims=True)
        return b / np.where(np.abs(total) < 1e-12, np.nan, total)
    error = np.square(after-bayes(before)).sum(-1)
    # Constant train-mean decoder, under the same test symbols.
    mean = flat(train_b).mean(0, keepdims=True)
    baseline = np.square(mean-bayes(np.broadcast_to(mean, before.shape))).sum(-1)
    return {"mse": float(np.nanmean(error)), "constant_decoder_mse": float(np.nanmean(baseline)),
            "invalid_simplex_fraction": float(np.mean((before < 0).any(-1) | (before > 1).any(-1))),
            "nonfinite_fraction": float(np.mean(~np.isfinite(error)))}


def matrix_error(spec, p, process, ladder):
    if spec.kind != "hmm":
        return None
    n = spec.k+1
    if n == len(process.initial):
        T, E = np.asarray(process.transitions), np.asarray(process.emission)
        label = "full"
        learned_e = np.asarray(jax.nn.softmax(p["E"], -1))
    elif ladder and n == 2:
        from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process
        _, T, E = quotient_model(ladder_process(2))
        fine_e = np.asarray(jax.nn.softmax(p["E"], -1))
        learned_e = np.stack((fine_e[:, :2].sum(-1), fine_e[:, 2]), -1)
        label = "exact_reward_state_quotient"
    else:
        return {"applicable": False, "reason": "different state counts"}
    learned_t = np.asarray(jax.nn.softmax(p["T"], -1))
    def score(order):
        dt = np.square(learned_t[:, order][:, :, order]-T).mean()
        de = np.square(learned_e[list(order)]-E).mean()
        return float(dt+de), float(dt), float(de)
    permutations = itertools.permutations(range(n))
    best_score, order = float("inf"), None
    while True:
        chunk = list(itertools.islice(permutations, 512))
        if not chunk:
            break
        orders = np.asarray(chunk)
        permuted_t = learned_t[:, orders[:, :, None], orders[:, None, :]].transpose(1, 0, 2, 3)
        errors = np.square(permuted_t-T).mean((1, 2, 3)) + np.square(learned_e[orders]-E).mean((1, 2))
        i = errors.argmin()
        if errors[i] < best_score:
            best_score, order = errors[i], chunk[i]
    method = "exhaustive"
    total, t, e = score(order)
    return {"applicable": True, "reference": label, "permutation_method": method,
            "permutation": list(map(int, order)), "transition_mse": t, "emission_mse": e,
            "combined_mse": total}
