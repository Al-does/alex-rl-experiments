import json

import pytest

from experiments.reward_factors_block2_steering_2026_11 import report, seed_campaign


@pytest.mark.parametrize("seed", (43, 44))
def test_approved_campaign_preserves_protocol(monkeypatch, tmp_path, seed):
    calls = []
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(seed_campaign, "__file__", str(tmp_path / "seed_campaign.py"))
    monkeypatch.setattr(seed_campaign.sys, "argv", ["seed_campaign", "--seed", str(seed)])
    monkeypatch.setattr(seed_campaign, "prepare_archives", lambda: None)
    monkeypatch.setattr(seed_campaign.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(report, "aggregate", lambda root, seeds: dict(seeds=list(seeds), files={}))
    seed_campaign.main()
    assert len(calls) == 4
    assert "--smoke" in calls[0] and all("--smoke" not in call for call in calls[1:])
    assert all(call[call.index("--seed") + 1] == str(seed) for call in calls)
    assert calls[1][calls[1].index("--site") + 1] == "1"
    assert calls[3][-7:] == ["--method", "eq4", "eq4_joint", "random", "action", "das", "map"]
    saved = json.loads((tmp_path / "results" / f"campaign_validation_seed{seed}.json").read_text())
    assert saved["seeds"] == [seed]


def test_campaign_rejects_unapproved_seed(monkeypatch):
    monkeypatch.setattr(seed_campaign.sys, "argv", ["seed_campaign", "--seed", "42"])
    with pytest.raises(SystemExit):
        seed_campaign.main()
