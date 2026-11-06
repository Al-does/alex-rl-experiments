from __future__ import annotations

import numpy as np
import pytest

from experiments.pusher_b.probe_analysis.effective_rank import ScatterAccumulator, spectrum_metrics


def test_spectrum_metrics_flat_and_rank_one():
    flat = spectrum_metrics(np.ones(16))
    assert flat["erank"] == pytest.approx(16.0)
    assert flat["participation_ratio"] == pytest.approx(16.0)
    assert flat["n90"] == 15 and flat["n99"] == 16
    one = spectrum_metrics(np.array([3.0, 0.0, 0.0]))
    assert one["erank"] == pytest.approx(1.0)
    assert one["participation_ratio"] == pytest.approx(1.0)
    assert one["n99"] == 1 and one["spectral_norm"] == 3.0


def test_scatter_accumulator_matches_direct_svd():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(500, 6)) @ np.diag([5.0, 3.0, 1.0, 0.2, 0.05, 0.0]) + 2.0
    acc = ScatterAccumulator(6)
    for chunk in np.array_split(rows, 7):
        acc.add(chunk)
    got = acc.metrics()
    centered = np.linalg.svd(rows - rows.mean(0), compute_uv=False)
    raw = np.linalg.svd(rows, compute_uv=False)
    assert got["centered_erank"] == pytest.approx(spectrum_metrics(centered)["erank"], rel=1e-6)
    assert got["uncentered_erank"] == pytest.approx(spectrum_metrics(raw)["erank"], rel=1e-6)
    assert got["centered_n99"] == spectrum_metrics(centered)["n99"]
    assert got["rms_norm"] == pytest.approx(np.sqrt((rows ** 2).sum(1).mean()))
    assert got["n_rows"] == 500
