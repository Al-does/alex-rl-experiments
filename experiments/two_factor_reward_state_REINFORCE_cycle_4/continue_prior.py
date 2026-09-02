"""Restore a completed cycle-4 run from B2 and continue training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from devops.serverless.retrieve import load_manifest, retrieve_manifest_artifacts
from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTINUATION_SPEC_FILENAME,
    CONTINUATION_TOTAL_ENV_STEPS,
    STEP_CHECKPOINT_INTERVAL,
)
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


def _load_artifact_manifest(results_dir: Path) -> dict:
    manifest_path = results_dir / "remote_artifacts.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())
    run_manifest_path = results_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(
            f"missing artifact manifest and run manifest under {results_dir}"
        )
    run_manifest = json.loads(run_manifest_path.read_text())
    remote = run_manifest.get("remote_artifacts") or {}
    manifest_key = remote.get("canonical_manifest_key")
    if not manifest_key:
        raise ValueError(
            f"run manifest {run_manifest_path} is missing remote_artifacts.canonical_manifest_key"
        )
    return load_manifest(key=str(manifest_key))


def _prior_agent_steps(tune_summary_path: Path) -> int:
    tune_summary = json.loads(tune_summary_path.read_text())
    trials = tune_summary.get("trials") or []
    if len(trials) != 1:
        raise RuntimeError(
            f"expected one prior trial in {tune_summary_path}, found {len(trials)}"
        )
    metrics = trials[0].get("metrics") or {}
    steps = metrics.get("env_runners/num_env_steps_sampled_lifetime")
    if steps is None:
        steps = metrics.get("num_env_steps_sampled_lifetime")
    if steps is None:
        raise RuntimeError(
            f"prior lifetime step count missing in {tune_summary_path}"
        )
    return int(steps)


def continue_prior_run(
    experiment_module: str,
    *,
    prior_run_id: str,
    seed: int = 42,
    upload_artifacts: bool = True,
    hardware_profile: str = "cuda4090",
) -> None:
    experiment = load_experiment(experiment_module)
    results_dir = experiment.directory / "results" / prior_run_id
    tune_summary_path = results_dir / "tune_summary.json"
    if not tune_summary_path.is_file():
        raise FileNotFoundError(f"missing tune summary: {tune_summary_path}")

    manifest = _load_artifact_manifest(results_dir)
    artifacts_root = experiment.directory / "artifacts" / prior_run_id
    retrieve_manifest_artifacts(manifest, artifacts_root)

    tune_summary = json.loads(tune_summary_path.read_text())
    trials = tune_summary.get("trials") or []
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

    prior_steps = _prior_agent_steps(tune_summary_path)
    target_steps = CONTINUATION_TOTAL_ENV_STEPS
    context = make_run_context(
        experiment,
        seed=seed,
        resume_from=resume_from,
        hardware_profile=hardware_profile,
    )
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (context.artifacts_dir / CONTINUATION_SPEC_FILENAME).write_text(
        json.dumps(
            {
                "prior_run_id": prior_run_id,
                "prior_agent_steps": prior_steps,
                "additional_agent_steps": target_steps - prior_steps,
                "target_agent_steps": target_steps,
                "step_checkpoint_interval": STEP_CHECKPOINT_INTERVAL,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    execute_experiment(
        experiment,
        context,
        command=[
            "python",
            "-m",
            "experiments.two_factor_reward_state_REINFORCE_cycle_4.continue_prior",
            experiment_module,
            "--prior-run-id",
            prior_run_id,
            "--seed",
            str(seed),
            "--hardware-profile",
            hardware_profile,
        ],
        runtime_overrides={
            "prior_run_id": prior_run_id,
            "resume_from": str(resume_from),
            "prior_agent_steps": prior_steps,
            "target_agent_steps": target_steps,
        },
        upload_artifacts=upload_artifacts,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a prior cycle-4 run's artifacts from B2 and continue "
            "training until CONTINUATION_TOTAL_ENV_STEPS lifetime env steps."
        )
    )
    parser.add_argument(
        "experiment_module",
        help="Importable leaf experiment module ending in .experiment",
    )
    parser.add_argument(
        "--prior-run-id",
        required=True,
        help="Run id of the completed checkpoint run (results/ and B2 prefix).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hardware-profile",
        default="cuda4090",
    )
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
        hardware_profile=args.hardware_profile,
    )


if __name__ == "__main__":
    main()
