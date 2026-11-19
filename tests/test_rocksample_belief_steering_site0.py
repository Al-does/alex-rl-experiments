import json
import sys

import numpy as np
import pytest

from experiments.rocksample_jax_2026_10 import belief_steering_site0 as v4


def test_frozen_replication_dispatch(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    for name in v4.HISTORICAL:
        directory = root / "results" / name
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps(dict(nuisance_features="frozen design")))
    for name in ("belief_probe.py", "model.py", "env.py"):
        (root / name).write_text("")
    monkeypatch.setattr(v4.bs.bp, "ROOT", root)
    monkeypatch.setattr(v4.bs.bp, "revision", lambda _: "test-revision")
    monkeypatch.setattr(v4.bs.bp, "checkpoint", lambda *a, **kw:
        (None, None, dict(d_model=8, n_layers=3, n_heads=2, context_len=2), {}))
    fitted = (np.zeros((4, 8, 7)), np.zeros((4, 7)),
              np.zeros((4, 7, 8)), np.zeros((4, 7, 8)))
    fits = []
    def maps(spec, env, params, episodes):
        fits.append(episodes)
        return fitted, [[0]*7]*4, fitted[2], {}
    monkeypatch.setattr(v4.bs, "maps", maps)
    monkeypatch.setattr(v4.bs, "calibrated_maps", lambda _: ({"calibrated_encoder": fitted}, []))
    calls = []
    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return []
    monkeypatch.setattr(v4.bs, "evaluate", evaluate)
    monkeypatch.setattr(v4.bs, "tune", lambda *a: pytest.fail("v4 must not tune"))
    monkeypatch.setattr(v4, "write_report", lambda *a: None)
    monkeypatch.setattr(v4, "figures", lambda *a: None)
    output = tmp_path / "v4"
    monkeypatch.setattr(sys, "argv", ["site0", "--output", str(output)])
    v4.main()
    report = json.loads((output / "results.json").read_text())
    assert fits == [256]*3
    assert [(a[4], a[5], a[6], a[7]) for a, _ in calls] == [(1024, 271829, 0, 1.)]*3
    assert report["seeds"] == dict(fit=41101, test=51203, null=81517, confirm=271829)
    assert len(report["agents"]) == 3
    assert all(kw["trace_dir"].is_dir() for _, kw in calls)


def test_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["site0", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        v4.main()


def test_empty_report_preserves_selection_disclosure(tmp_path):
    v4.write_report(tmp_path, dict(protocol=v4.PROTOCOL, nuisance_features="frozen", agents=[]))
    text = (tmp_path / "findings.md").read_text()
    assert "after looking at v3 confirmation data" in text
    assert "No tuning or selection on v4 data" in text
    assert "episode 0 for both directions was fixed" in text
