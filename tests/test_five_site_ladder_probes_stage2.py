import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes_stage2 as stage2


def test_bootstrap_mean_ci_constant_and_known_resamples():
    indices = np.stack([np.zeros(15, dtype=int), np.full(15, 14, dtype=int)])
    assert stage2.bootstrap_mean_ci(np.full(15, 0.25), indices) == {
        "mean": 0.25, "ci95_low": 0.25, "ci95_high": 0.25}
    actual = stage2.bootstrap_mean_ci(np.arange(15, dtype=float), indices)
    assert actual == pytest.approx({"mean": 7.0, "ci95_low": 0.35, "ci95_high": 13.65})


@pytest.mark.parametrize("values", [[], [1.0], [np.nan] * 15, [np.inf] * 15])
def test_bootstrap_mean_ci_rejects_missing_or_nonfinite_seeds(values):
    with pytest.raises(ValueError, match="15 finite"):
        stage2.bootstrap_mean_ci(values, np.zeros((2, 15), dtype=int))


def test_summary_rejects_empty_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(stage2, "OUTPUT", tmp_path)
    monkeypatch.setattr(stage2, "stage1_hashes", lambda: {})
    (tmp_path / "stage1_snapshot.json").write_text("{}")
    with pytest.raises(ValueError, match="complete 90-report grid"):
        stage2.summarize()
    assert not (tmp_path / "summary.json").exists()


def test_summary_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(stage2, "OUTPUT", tmp_path)
    path = tmp_path / "summary.json"
    path.write_text("preserved")
    with pytest.raises(FileExistsError):
        stage2.summarize()
    assert path.read_text() == "preserved"
