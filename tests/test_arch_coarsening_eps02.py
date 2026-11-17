import hashlib
import json

import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_coarsening import analyze
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment


def process(variant):
    p = environment.ladder_process(variant)
    return {"initial": np.asarray(p.initial, dtype=np.float64),
            "emission": np.asarray(p.emission, dtype=np.float64),
            "transitions": np.asarray(p.transitions, dtype=np.float64)}


@pytest.mark.parametrize("variant", (1, 2))
def test_merged_matches_exact_two_state_filter(variant):
    p = process(variant)
    tokens = np.array([[0, 2, 1, 2, 2], [1, 2, 0, 2, 0]])
    actions = np.array([[0, 1, 0, 2, 1], [0, 0, 2, 1, 2]])
    obs = np.concatenate((np.eye(3)[tokens], np.eye(3)[actions]), axis=-1)
    obs[:, 0, 3:] = 0
    actual = analyze.merged_filter(obs, p)[..., 0]
    emission = np.stack((p["emission"][:, :2].sum(1), p["emission"][:, 2]), -1)[[0, 2]]
    trans = np.stack((p["transitions"][:, :, :2].sum(2), p["transitions"][:, :, 2]), -1)[:, [0, 2]]
    b = np.array([p["initial"][:2].sum(), p["initial"][2]], dtype=np.float64)
    b = np.broadcast_to(b, (2, 2)).copy()
    for t in range(5):
        if t:
            b = np.einsum("bi,bij->bj", b, trans[actions[:, t]])
        b *= emission[:, (tokens[:, t] == 2).astype(int)].T
        b /= b.sum(1, keepdims=True)
        np.testing.assert_allclose(actual[:, t], b[:, 1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual[:, 0], actual[0, 0])


def test_full_state_merged_control_and_token_invariance():
    p = process(3)
    obs = np.concatenate((np.eye(3)[[[0, 2, 1, 2]]], np.eye(3)[[[0, 1, 2, 0]]]), -1)
    alternate = obs.copy()
    alternate[..., :2] = obs[..., 1::-1]
    result = analyze.merged_filter(obs, p)
    np.testing.assert_array_equal(result, analyze.merged_filter(alternate, p))
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.parametrize("shape", ((0, 4, 6), (2, 0, 6), (2, 4, 5), (4, 6)))
def test_invalid_or_empty_histories(shape):
    with pytest.raises(ValueError, match="nonempty"):
        analyze.merged_filter(np.zeros(shape), process(2))


def test_impossible_history():
    p = process(2)
    p["emission"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="impossible"):
        analyze.merged_filter(np.zeros((1, 1, 6)), p)


def test_affine_score_and_warmup():
    rng = np.random.default_rng(42)
    x, z = rng.normal(size=(100, 4)), rng.normal(size=(80, 4))
    w = rng.normal(size=(4, 3))
    assert analyze.score(x, z, x @ w + 2, z @ w + 2)["error"] < 1e-12
    data = np.arange(2 * 70).reshape(2, 70, 1)
    np.testing.assert_array_equal(analyze.flatten(data, 8)[:, 0], [64, 65, 66, 67, 68, 69, 134, 135])


@pytest.mark.parametrize("corrupt", (False, True))
def test_checkpoint_integrity(tmp_path, monkeypatch, corrupt):
    from harness.storage.b2 import B2StorageConfig
    directory = tmp_path / "results" / "run"
    directory.mkdir(parents=True)
    (directory / "run_manifest.json").write_text(json.dumps({"remote_artifacts": {
        "bucket": "test", "canonical_manifest_key": "manifest"}}))
    payload = b"trusted checkpoint"
    row = {"relative_path": "checkpoints/step_0000000001.npz", "key": "checkpoint",
           "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    class Client:
        def get_object(self, **kwargs):
            from io import BytesIO
            return {"Body": BytesIO(json.dumps({"files": [row]}).encode())}

        def download_file(self, bucket, key, path):
            from pathlib import Path
            Path(path).write_bytes(b"corrupt" if corrupt else payload)

    monkeypatch.setattr(B2StorageConfig, "from_env", lambda: type("Config", (), {"s3_client": lambda self: Client()})())
    monkeypatch.setattr(analyze, "STUDY", tmp_path)
    if corrupt:
        with pytest.raises(ValueError, match="verification failed"):
            analyze.verified_checkpoint(directory, 1)
    else:
        path, provenance = analyze.verified_checkpoint(directory, 1)
        assert path.read_bytes() == payload
        assert provenance["checkpoint_sha256"] == row["sha256"]
