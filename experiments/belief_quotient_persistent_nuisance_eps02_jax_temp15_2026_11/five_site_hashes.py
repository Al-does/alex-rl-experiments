"""Hash data files on disk and archived sources at their recorded Git revisions."""
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes as probe

SOURCE_REVISIONS = "source_revisions.json"


def source_digest(study, name, revision):
    path = str((study / name).relative_to(probe.ROOT))
    return hashlib.sha256(probe.git_text(revision, path).encode()).hexdigest()


def verify_manifest(path, study=None):
    study = probe.STUDY if study is None else study
    manifest = json.loads(path.read_text())
    metadata = path.with_name(SOURCE_REVISIONS)
    revisions = json.loads(metadata.read_text()) if metadata.exists() else {}
    for name, expected in manifest.items():
        if Path(name).suffix == ".py":
            if name not in revisions:
                raise ValueError(f"missing source revision: {name}")
            actual = source_digest(study, name, revisions[name])
        else:
            actual = hashlib.sha256((study / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"saved hash differs: {name}")
    return manifest


def write_manifest(path, files, study=None):
    study = probe.STUDY if study is None else study
    manifest = json.loads(path.read_text()) if path.exists() else {}
    metadata = path.with_name(SOURCE_REVISIONS)
    revisions = json.loads(metadata.read_text()) if metadata.exists() else {}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=probe.ROOT).decode().strip()
    for file in files:
        name = str(file.relative_to(study))
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if file.suffix == ".py":
            if source_digest(study, name, revision) != digest:
                raise ValueError(f"source differs from HEAD; commit before recording hashes: {name}")
            revisions[name] = revision
        manifest[name] = digest
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    metadata.write_text(json.dumps(revisions, indent=2) + "\n")
