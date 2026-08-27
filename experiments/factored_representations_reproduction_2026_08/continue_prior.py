"""Restore a completed 5M run from B2 and continue training to 50M steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from devops.serverless.retrieve import retrieve_manifest_artifacts
from harness.cli import execute_experiment, load_experiment, make_run_context


def _resume_checkpoint(
    *,
    experiment_dir: Path,
    prior_run_id: str,
    checkpoint_remote: str,
) -> Path:
    marker = f"/artifacts/{prior_run_id}/"
    if marker not in checkpoint_remote:
        raise ValueError(
            "tune_summary checkpoint path does not match prior_run_id "
            f"{prior_run_id!r}"
        )
    relative = checkpoint_remote.split(marker, 1)[1]
    return experiment_dir / "artifacts" / prior_run_id / relative


def continue_prior_run(
    experiment_module: str,
    *,
    prior_run_id: str,
    seed: int = 42,
    upload_artifacts: bool = True,
) -> None:
    experiment = load_experiment(experiment_module)
    results_dir = experiment.directory / "results" / prior_run_id
    manifest_path = results_dir / "remote_artifacts.json"
    tune_summary_path = results_dir / "tune_summary.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing artifact manifest: {manifest_path}")
    if not tune_summary_path.is_file():
        raise FileNotFoundError(f"missing tune summary: {tune_summary_path}")

    manifest = json.loads(manifest_path.read_text())
    artifacts_root = experiment.directory / "artifacts" / prior_run_id
    retrieve_manifest_artifacts(manifest, artifacts_root)

    tune_summary = json.loads(tune_summary_path.read_text())
    trials = tune_summary.get("trials") or []
    if len(trials) != 1:
        raise RuntimeError(
            f"expected one prior trial in {tune_summary_path}, found {len(trials)}"
        )
    checkpoint_remote = str(trials[0].get("checkpoint") or "")
    if not checkpoint_remote:
        raise RuntimeError(f"prior trial checkpoint missing in {tune_summary_path}")

    resume_from = _resume_checkpoint(
        experiment_dir=experiment.directory,
        prior_run_id=prior_run_id,
        checkpoint_remote=checkpoint_remote,
    )
    if not resume_from.is_dir():
        raise FileNotFoundError(
            f"restored checkpoint directory not found: {resume_from}"
        )

    context = make_run_context(
        experiment,
        seed=seed,
        resume_from=resume_from,
    )
    execute_experiment(
        experiment,
        context,
        command=[
            "python",
            "-m",
            "experiments.factored_representations_reproduction_2026_08.continue_prior",
            experiment_module,
            "--prior-run-id",
            prior_run_id,
            "--seed",
            str(seed),
        ],
        runtime_overrides={
            "prior_run_id": prior_run_id,
            "resume_from": str(resume_from),
        },
        upload_artifacts=upload_artifacts,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a prior run's artifacts from B2 and continue training "
            "to the current TOTAL_ENV_STEPS budget."
        )
    )
    parser.add_argument(
        "experiment_module",
        help="Importable leaf experiment module ending in .experiment",
    )
    parser.add_argument(
        "--prior-run-id",
        required=True,
        help="Run id of the completed 5M checkpoint run (results/ and B2 prefix).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    continue_prior_run(
        args.experiment_module,
        prior_run_id=args.prior_run_id,
        seed=args.seed,
        upload_artifacts=args.upload_artifacts,
    )


if __name__ == "__main__":
    main()
