import json

import pytest

from experiments.rocksample_jax_2026_10 import belief_steering_continuation as cont


def fixture_source(tmp_path):
    source = dict(run_id="original", seed_index=2, env_steps=30_408_704,
                  sha256="digest", size_bytes=123, uri="s3://bucket/source")
    path = tmp_path/"continued"
    path.mkdir()
    (path/"continuation_source.json").write_text(json.dumps(source))
    historical = dict(agents=[dict(run_id="original", seed=2, checkpoint=source.copy())])
    return source, historical


def test_mapping_uses_source_record_not_run_order(tmp_path):
    source, historical = fixture_source(tmp_path)
    result = cont.mappings(tmp_path, historical)
    assert result[0]["continuation_run"] == "continued"
    assert result[0]["source"] == source
    assert result[0]["seed"] == 2


def test_mapping_missing_and_duplicate(tmp_path):
    with pytest.raises(ValueError, match="expected one continuation"):
        cont.mappings(tmp_path, dict(agents=[dict(run_id="missing", seed=2)]))
    source, historical = fixture_source(tmp_path)
    duplicate = tmp_path/"duplicate"
    duplicate.mkdir()
    (duplicate/"continuation_source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="expected one continuation"):
        cont.mappings(tmp_path, historical)


@pytest.mark.parametrize("key,value", [("sha256", "wrong"), ("env_steps", 1)])
def test_mapping_rejects_wrong_checkpoint(tmp_path, key, value):
    source, historical = fixture_source(tmp_path)
    source[key] = value
    (tmp_path/"continued/continuation_source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError):
        cont.mappings(tmp_path, historical)


def test_reuse_requires_both_direction_strengths():
    def tuning(good, bad):
        return dict(representations={"joint": {
            "Good→Bad": dict(strength=good), "Bad→Good": dict(strength=bad)}})
    assert cont.same_strengths(tuning(1, 1), tuning(1, 1), "joint")
    assert not cont.same_strengths(tuning(.5, 1), tuning(1, 1), "joint")
    assert not cont.same_strengths(tuning(1, .5), tuning(1, 1), "joint")


def test_loader_rejects_noncanonical_layout_before_checkpoint(tmp_path, monkeypatch):
    run = tmp_path/"ppo_bmax_r7_cont330m/results/continued"
    run.mkdir(parents=True)
    (run/"run_manifest.json").write_text(json.dumps(dict(remote_artifacts={
        "prefix": "wrong", "canonical_manifest_key": "wrong"})))
    monkeypatch.setattr(cont, "ROOT", tmp_path)
    monkeypatch.setattr(cont.bs.bp, "checkpoint", lambda *a, **kw: pytest.fail("must not unpickle"))
    with pytest.raises(ValueError, match="canonical durability"):
        cont.load(dict(continuation_run="continued", seed=2), 316)
