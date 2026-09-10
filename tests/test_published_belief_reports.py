import hashlib
import json
from pathlib import Path

import pytest

from experiments import gol_results_report as gol
from experiments import strata_results_report as strata


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {gol: gol.OUTPUT, strata: strata.OUTPUT}
REPORT_FILES = {gol: "saved_results.json", strata: "summary.json"}


def _load(path):
    return json.loads(path.read_text())


def _hash(path):
    path = path.resolve(strict=True)
    assert path.is_relative_to(ROOT)
    assert "artifacts" not in path.parts
    assert path.suffix in {".json", ".jsonl", ".csv", ".png", ".md"}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(directory):
    return {path.relative_to(directory): _hash(path) for path in directory.rglob("*") if path.is_file()}


def _unexpected(*args, **kwargs):
    pytest.fail("Unexpected extraction, numerical reference calculation, plotting, or write")


@pytest.fixture(params=[gol, strata], ids=["gol", "strata"])
def report(request, monkeypatch):
    module = request.param
    monkeypatch.setattr(module, "OUTPUT", ARCHIVES[module])
    monkeypatch.setattr(module, "plots" if module is gol else "plot", _unexpected)
    return module


def test_default_report_directory_is_read_only(report, monkeypatch, capsys):
    before = _snapshot(report.OUTPUT)
    assert before
    monkeypatch.setattr(report, "extract", _unexpected)
    with pytest.raises(SystemExit) as error:
        report.main([])
    assert error.value.code == 2
    assert "--output-dir NEW_DIRECTORY" in capsys.readouterr().err
    assert _snapshot(report.OUTPUT) == before


@pytest.mark.parametrize("kind", ["directory", "empty_directory", "file", "symlink", "dangling_symlink"])
def test_existing_output_path_is_rejected(report, kind, tmp_path, monkeypatch, capsys):
    protected = tmp_path / "protected"
    protected.mkdir()
    sentinel = protected / "sentinel"
    sentinel.write_bytes(b"published bytes\x00\xff")
    output = tmp_path / "report"
    if kind == "directory":
        output = protected
    elif kind == "empty_directory":
        output.mkdir()
    elif kind == "file":
        output = sentinel
    else:
        output.symlink_to(protected if kind == "symlink" else tmp_path / "missing", target_is_directory=True)
    monkeypatch.setattr(report, "extract", _unexpected)
    with pytest.raises(SystemExit) as error:
        report.main(["--output-dir", str(output), "--no-plots"])
    assert error.value.code == 2
    assert "--output-dir NEW_DIRECTORY" in capsys.readouterr().err
    assert sentinel.read_bytes() == b"published bytes\x00\xff"
    assert list(protected.iterdir()) == [sentinel]
    if kind == "empty_directory":
        assert not list(output.iterdir())


def test_validation_failure_does_not_create_output(report, tmp_path, monkeypatch):
    output = tmp_path / "new-parent" / "report"

    def invalid(*args):
        assert not output.parent.exists()
        raise ValueError("Invalid compact input")

    monkeypatch.setattr(report, "extract" if report is gol else "numerical_verification", invalid)
    with pytest.raises(ValueError, match="Invalid compact input"):
        report.main(["--output-dir", str(output), "--no-plots"])
    assert not output.parent.exists()


def test_output_creation_race_does_not_overwrite(report, tmp_path, monkeypatch):
    output = tmp_path / "report"
    archived = _load(ARCHIVES[report] / REPORT_FILES[report])

    def competing_writer(*args):
        assert not output.exists()
        output.mkdir()
        (output / "sentinel").write_bytes(b"other publisher")
        return {} if report is gol else archived["numerical_verification"]

    monkeypatch.setattr(report, "provenance" if report is gol else "numerical_verification", competing_writer)
    monkeypatch.setattr(report, "write_json" if report is gol else "dump_csv", _unexpected)
    with pytest.raises(FileExistsError):
        report.main(["--output-dir", str(output), "--no-plots"])
    assert [path.name for path in output.iterdir()] == ["sentinel"]
    assert (output / "sentinel").read_bytes() == b"other publisher"


def test_compact_report_regeneration_preserves_archive(report, tmp_path, monkeypatch):
    archive = ARCHIVES[report]
    before = _snapshot(archive)
    archived = _load(archive / REPORT_FILES[report])
    output = tmp_path / "new-parent" / "report"
    if report is strata:
        def saved_verification(recipes):
            assert not output.parent.exists()
            assert set(recipes) == set(strata.RUNS)
            return archived["numerical_verification"]

        monkeypatch.setattr(strata, "numerical_verification", saved_verification)
        monkeypatch.setattr(strata, "token_references", _unexpected)
    report.main(["--output-dir", str(output), "--no-plots"])
    generated = _load(output / REPORT_FILES[report])
    metadata = {"generator", "experiment_checkout" if report is gol else "analysis_commit"}
    assert {key: value for key, value in generated.items() if key not in metadata} == {
        key: value for key, value in archived.items() if key not in metadata
    }
    tables = sorted(archive.glob("*.csv"))
    assert len(tables) == (5 if report is gol else 9)
    assert {path.name for path in output.glob("*.csv")} == {path.name for path in tables}
    for table in tables:
        assert (output / table.name).read_bytes() == table.read_bytes(), table.name
    assert not list(output.glob("*.png"))
    assert _snapshot(archive) == before


def test_archived_report_source_hashes(report):
    archived = _load(ARCHIVES[report] / REPORT_FILES[report])
    sources = archived["sources"]
    path_key = "source" if report is gol else "path"
    assert len(sources) == (59 if report is gol else 63)
    assert len({row[path_key] for row in sources}) == len(sources)
    for row in sources:
        path = Path(row[path_key])
        assert not path.is_absolute()
        assert _hash(ROOT / path) == row["sha256"], row[path_key]
    if report is strata:
        reference = archived["numerical_verification"]["token_cycle_2"]["renewal_refinement_source"]
        assert _hash(ROOT / reference["path"]) == reference["sha256"]


def test_reviewed_strata_output_manifest_hashes():
    directory = ROOT / "experiments/strata_token_guess_cycle_1/ppo/results/belief_geometry_reviewed_20260909"
    outputs = _load(directory / "output_manifest.json")["outputs_sha256"]
    assert len(outputs) == 8
    for name, expected in outputs.items():
        path = directory / name
        assert path.resolve().is_relative_to(directory.resolve())
        assert _hash(path) == expected, name


@pytest.mark.parametrize("location", [
    "wing_token_guess_cycle_1/ppo/results/probe_controls_20260908",
    "wing_token_guess_cycle_2/ppo/results/probe_simplex",
])
def test_wing_simplex_png_matches_json_sidecar(location):
    sidecar = ROOT / "experiments" / location / "belief_simplex.json"
    assert _hash(sidecar.with_suffix(".png")) == _load(sidecar)["png_sha256"]


def test_strata_legacy_output_alias_is_guarded(tmp_path, monkeypatch):
    monkeypatch.setattr(strata, "extract", _unexpected)
    with pytest.raises(SystemExit) as error:
        strata.main(["--output", str(tmp_path)])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
