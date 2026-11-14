import json

import pytest

from experiments.belief_quotient_history_jax_lumped_temp15_2026_11 import plot_reprobe


def test_load_reports_complete_and_missing_targets(tmp_path):
    for variant in (1, 2, 3):
        for architecture in plot_reprobe.ARCHITECTURES:
            targets = set(plot_reprobe.TARGET_DEFINITIONS)
            if variant == 3:
                targets.remove("coarse_reward_class")
            rows = [{"checkpoint_step": i, "targets": {t: {} for t in targets}} for i in range(202)]
            path = tmp_path / f"variant_{variant}_{architecture}" / "results" / "run" / "reprobe_targets.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"smoke": False, "checkpoint_reports": rows}))
    reports, hashes = plot_reprobe.load_reports(tmp_path)
    assert len(reports) == len(hashes) == 12
    rows[0]["targets"].pop("belief")
    path.write_text(json.dumps({"smoke": False, "checkpoint_reports": rows}))
    with pytest.raises(ValueError, match="wrong targets"):
        plot_reprobe.load_reports(tmp_path)


def test_load_reports_empty(tmp_path):
    with pytest.raises(ValueError, match="expected one report"):
        plot_reprobe.load_reports(tmp_path)
