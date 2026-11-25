"""Fixed alpha-vector references from the original critic-aware ladder probe."""

from dataclasses import dataclass
import hashlib
import io
from zipfile import ZipFile

import numpy as np

from analysis.probes.linear import fit_affine_probe


PBVI_SHA = {
    1: "b18261128f60ef083e5c95775564affaa6ddeac6a8e466c05d1fd2924e0f21b1",
    2: "6e58bececa51dadfe9a2144159859f6ec6b2c084bfd20b3892d89d5d5ad1d797",
    3: "904fb571a9ec033e1852c9aafd732fb940eb9f863251a3113ec3d8ce115c0d2e",
}


@dataclass
class ValueReference:
    alphas: tuple[np.ndarray, ...]
    metadata: dict

    def q_values(self, belief):
        b = np.asarray(belief, dtype=np.float64)
        # Limit the temporary rows-by-alpha array.
        groups = np.array_split(b, max(1, (len(b) + 2047) // 2048))
        return np.concatenate([
            np.column_stack([(group @ alphas.T).max(axis=1) for alphas in self.alphas])
            for group in groups
        ])


def restore_reference(path, variant, process, gamma):
    with ZipFile(path) as archive:
        data = archive.read(f"pbvi_variant_{variant}.npz")
    digest = hashlib.sha256(data).hexdigest()
    if digest != PBVI_SHA[variant]:
        raise ValueError(f"PBVI reference integrity mismatch for {variant=}")
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        for name in ("transitions", "emission"):
            np.testing.assert_allclose(saved[name], getattr(process, name), rtol=0, atol=1e-7)
        np.testing.assert_allclose(saved["reward"][0], process.reward, rtol=0, atol=0)
        np.testing.assert_allclose(saved["gamma"], gamma, rtol=0, atol=0)
        alphas = tuple(saved[f"alpha_action_{a}"] for a in range(int(saved["n_actions"])))
        metadata = {
            "sha256": digest,
            "bellman_residual_max_on_grid": float(saved["bellman_residual_max"]),
            "alpha_vectors": sum(len(x) for x in alphas),
        }
    return ValueReference(alphas, metadata)


def build_targets(reference, beliefs):
    q_fit, q_test = map(reference.q_values, beliefs)
    v_fit, v_test = q_fit.max(1, keepdims=True), q_test.max(1, keepdims=True)
    plain = {"q_star": (q_fit, q_test), "v_star": (v_fit, v_test),
             "a_star": (q_fit - v_fit, q_test - v_test)}
    results = plain.copy()
    variance_shares = {}
    designs = [np.column_stack([b.astype(np.float64), np.ones(len(b))]) for b in beliefs]
    for name, (train, test) in plain.items():
        weight, *_ = np.linalg.lstsq(designs[0], train, rcond=1e-12)
        residual = (train - designs[0] @ weight, test - designs[1] @ weight)
        results[f"{name}_residual"] = residual
        variance_shares[name] = float(residual[1].var(axis=0).sum() / test.var(axis=0).sum())
    return results, variance_shares


def score_targets(features, values):
    # One regression for all targets; ridge is identical across columns.
    training = np.concatenate([v[0] for v in values.values()], axis=1)
    weight, bias = fit_affine_probe(features[0], training, ridge=1e-6)
    predictions = features[1] @ weight + bias
    results = {}
    offset = 0
    for name, (_, test) in values.items():
        predicted = predictions[:, offset:offset + test.shape[1]]
        offset += test.shape[1]
        variance = float(test.var(axis=0).sum())
        mse = float(np.square(test - predicted).sum(axis=1).mean())
        results[name] = {"error": mse / variance if variance >= 1e-12 else None,
                         "target_variance": variance, "mse": mse}
    return results
