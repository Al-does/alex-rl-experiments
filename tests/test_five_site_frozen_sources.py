import hashlib
import subprocess

import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes as probe


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def shallow_source(tmp_path):
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "init", "-q", "-b", "main", str(upstream)], check=True)
    source = "VALUE = 42\n"
    (upstream / "frozen.py").write_text(source)
    (upstream / "empty.py").write_text("")
    git(upstream, "add", "frozen.py", "empty.py")
    commit = ("-c", "user.name=Probe test", "-c", "user.email=probe-test@example.invalid", "commit", "-qm")
    git(upstream, *commit, "pinned source")
    revision = git(upstream, "rev-parse", "HEAD")
    (upstream / "frozen.py").write_text("VALUE = 99\n")
    git(upstream, "add", "frozen.py")
    git(upstream, *commit, "new source")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--depth=1", "--single-branch",
                    upstream.as_uri(), str(clone)], check=True)
    assert git(clone, "rev-parse", "--is-shallow-repository") == "true"
    missing = subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                             cwd=clone, capture_output=True)
    assert missing.returncode != 0
    return upstream, clone, revision, source


@pytest.mark.parametrize("path", ["frozen.py", "empty.py"])
def test_frozen_module_fetches_exact_revision_in_shallow_clone(shallow_source, monkeypatch, path):
    _, clone, revision, source = shallow_source
    head = git(clone, "rev-parse", "HEAD")
    monkeypatch.setattr(probe, "ROOT", clone)
    module, digest = probe.frozen_module(revision, path)
    expected = source if path == "frozen.py" else ""
    assert digest == hashlib.sha256(expected.encode()).hexdigest()
    if path == "frozen.py":
        assert module.VALUE == 42
    assert git(clone, "rev-parse", "HEAD") == head
    assert git(clone, "cat-file", "-t", revision) == "commit"


@pytest.mark.parametrize("path", ["frozen.py", "absent.py"])
def test_existing_revision_never_fetches_even_for_missing_path(shallow_source, monkeypatch, path):
    upstream, _, revision, source = shallow_source
    monkeypatch.setattr(probe, "ROOT", upstream)
    run = subprocess.run

    def reject_fetch(command, *args, **kwargs):
        assert command[:2] != ["git", "fetch"]
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", reject_fetch)
    if path == "absent.py":
        with pytest.raises(subprocess.CalledProcessError):
            probe.git_text(revision, path)
    else:
        assert probe.git_text(revision, path) == source


def test_missing_revision_fetch_failure_has_recovery_command(shallow_source, monkeypatch):
    _, clone, revision, _ = shallow_source
    git(clone, "remote", "remove", "origin")
    monkeypatch.setattr(probe, "ROOT", clone)
    with pytest.raises(RuntimeError, match="could not be fetched from origin") as error:
        probe.frozen_module(revision, "frozen.py")
    assert f"git fetch --no-tags --depth=1 origin {revision}" in str(error.value)
    assert str(clone) in str(error.value)
