import hashlib
import json
import subprocess

import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_hashes as hashes
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes_stage2 as stage2


@pytest.mark.parametrize("bundle,name", [
    ("five_site_probes_stage1_seeds42_44", "hashes.json"),
    ("five_site_probes_seeds42_56", "hashes.json"),
    ("five_site_probes_seeds42_56", "corrected_hashes.json"),
])
def test_archived_manifests_match_recorded_blobs_without_rewriting_digests(bundle, name):
    path = hashes.probe.STUDY / "results" / bundle / name
    manifest = hashes.verify_manifest(path)
    archived = subprocess.check_output(["git", "show", f"60fffae:{path.relative_to(hashes.probe.ROOT)}"],
                                       cwd=hashes.probe.ROOT)
    assert path.read_bytes() == archived
    revisions = json.loads(path.with_name(hashes.SOURCE_REVISIONS).read_text())
    assert all(revisions[name] == "60fffae92a1682420117b6731084918c2d9a0f87"
               for name in manifest if name.endswith(".py"))


def test_stage1_metadata_does_not_change_archived_snapshot():
    snapshot = json.loads((stage2.OUTPUT / "stage1_snapshot.json").read_text())
    assert stage2.stage1_hashes() == snapshot


@pytest.fixture
def saved_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(hashes.probe, "ROOT", tmp_path)
    study = tmp_path / "study"
    study.mkdir()
    (study / "adapter.py").write_text("changed working copy\n")
    (study / "data.json").write_text("{}")
    manifest = {"adapter.py": hashlib.sha256(b"original source\n").hexdigest(),
                "data.json": hashlib.sha256(b"{}").hexdigest()}
    path = study / "hashes.json"
    path.write_text(json.dumps(manifest))
    path.with_name(hashes.SOURCE_REVISIONS).write_text(json.dumps({"adapter.py": "old-revision"}))

    def git_text(revision, source):
        assert source == "study/adapter.py" and revision == "old-revision"
        return "original source\n"

    monkeypatch.setattr(hashes.probe, "git_text", git_text)
    return path, study, manifest


def test_verification_uses_recorded_blob_not_working_source(saved_manifest):
    path, study, manifest = saved_manifest
    assert hashes.verify_manifest(path, study) == manifest


@pytest.mark.parametrize("mode", ["missing_revision", "source_digest", "data_digest"])
def test_verification_rejects_missing_revision_or_tampering(saved_manifest, mode):
    path, study, manifest = saved_manifest
    if mode == "missing_revision":
        path.with_name(hashes.SOURCE_REVISIONS).unlink()
    else:
        manifest["adapter.py" if mode == "source_digest" else "data.json"] = "wrong"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing source revision|saved hash differs"):
        hashes.verify_manifest(path, study)


def test_empty_manifest_needs_no_source_metadata(tmp_path):
    path = tmp_path / "hashes.json"
    path.write_text("{}")
    assert hashes.verify_manifest(path, tmp_path) == {}


def test_write_manifest_preserves_archived_sources_and_records_new_source(saved_manifest, monkeypatch):
    path, study, manifest = saved_manifest
    monkeypatch.setattr(hashes.subprocess, "check_output", lambda *args, **kwargs: b"new-revision\n")
    original = hashes.probe.git_text

    def git_text(revision, source):
        if source == "study/new.py":
            assert revision == "new-revision"
            return "new source\n"
        return original(revision, source)

    monkeypatch.setattr(hashes.probe, "git_text", git_text)
    new = study / "new.py"
    new.write_text("new source\n")
    hashes.write_manifest(path, [new], study)
    revisions = json.loads(path.with_name(hashes.SOURCE_REVISIONS).read_text())
    assert revisions == {"adapter.py": "old-revision", "new.py": "new-revision"}
    assert hashes.verify_manifest(path, study)["adapter.py"] == manifest["adapter.py"]


def test_write_manifest_rejects_uncommitted_source(saved_manifest, monkeypatch):
    path, study, _ = saved_manifest
    before = path.read_bytes()
    monkeypatch.setattr(hashes.subprocess, "check_output", lambda *args, **kwargs: b"old-revision\n")
    with pytest.raises(ValueError, match="source differs from HEAD"):
        hashes.write_manifest(path, [study / "adapter.py"], study)
    assert path.read_bytes() == before
