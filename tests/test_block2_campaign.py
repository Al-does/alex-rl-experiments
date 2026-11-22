import json

import pytest

from experiments.reward_factors_block2_steering_2026_11 import campaign_report, report, seed_campaign


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


def test_sign_counts_are_strict_and_directional():
    assert campaign_report.signs([-1., 0., 2.], lower=True) == dict(wins=1, ties=1, losses=1, differences=[-1., 0., 2.])
    assert campaign_report.signs([1e-14, 0., 2.])["wins"] == 2
    for values in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            campaign_report.signs(values)
    for seeds in ([], [42, 42]):
        with pytest.raises(ValueError):
            campaign_report.derive(dict(seeds=seeds))
        with pytest.raises(ValueError):
            report.aggregate(None, seeds=seeds)


def test_campaign_r2_sample_sd_and_full_mirror_are_nondirectional():
    import statistics
    seeds = [42, 43, 44]
    per_seed, encoder_r2 = {}, {}
    for index, seed in enumerate(seeds):
        for site in (1, 3):
            value = float(index+1) if site == 1 else 2.
            for method in (*report.METHODS, *report.ENCODERS):
                per_seed[f"{seed}:{site}:{method}"] = dict.fromkeys(campaign_report.METRICS, value)
            for test in report.TESTS:
                for representation in (("marginal",) if test.startswith("ladder") else ("marginal", "joint")):
                    for kind in ("affine", "mlp"):
                        for source in ("exact", "decoded"):
                            encoder_r2[f"{seed}:{site}:{test}:{representation}_{kind}_{source}"] = value
    result = dict(seeds=seeds, per_seed=per_seed, encoder_r2=encoder_r2)
    derived = campaign_report.derive(result)
    assert derived["sign_counts"]["eq4"]["ladder_occupancy"]["wins"] == 1
    assert derived["sign_counts"]["eq4"]["full_mirror"] == dict(lower=1, ties=1, higher=1, differences=[-1., 0., 1.], direction="nondirectional")
    assert derived["encoder_r2_aggregate"]["1:ladder:marginal_affine_exact"] == dict(mean=2., sd=statistics.stdev([1., 2., 3.]))
    assert derived["encoder_r2_sign_counts"]["ladder:marginal_affine_exact"]["ties"] == 1
    result.update(derived)
    result["aggregate"] = {f"{site}:{method}": dict.fromkeys(campaign_report.METRICS, dict(mean=2., sd=1.))
                           for site in (1, 3) for method in (*report.METHODS, *report.ENCODERS)}
    text = campaign_report.tables(result)
    assert "Mean ± sample SD" in text and "42, 43, 44" in text
    assert "2.0000±1.0000" in text and "Full mirror < / = / >" in text


def test_audit_digest_and_invalid_output_label(monkeypatch):
    import hashlib
    import sys
    import numpy as np
    from experiments.reward_factors_block2_steering_2026_11 import control_audit, control_swap
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert control_audit.digest(values) == control_audit.digest(np.asfortranarray(values))
    assert control_audit.digest(values) != control_audit.digest(values+1)
    assert control_audit.digest(np.array([])) == hashlib.sha256(b"").hexdigest()
    monkeypatch.setattr(sys, "argv", ["control_swap", "--label", "../bad"])
    with pytest.raises(SystemExit):
        control_swap.main()
