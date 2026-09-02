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


def _prior_agent_steps_from_tune(tune_summary_path: Path) -> int:
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


def _prior_agent_steps(results_dir: Path) -> int:
    tune_summary_path = results_dir / "tune_summary.json"
    if tune_summary_path.is_file():
        return _prior_agent_steps_from_tune(tune_summary_path)
    summary_path = results_dir / "condition_summary.json"
    if summary_path.is_file():
        reports = json.loads(summary_path.read_text()).get("checkpoint_reports") or []
        if not reports:
            raise RuntimeError(f"no checkpoint reports in {summary_path}")
        return int(max(int(report["agent_steps"]) for report in reports))
    raise FileNotFoundError(
        f"missing tune summary or condition summary under {results_dir}"
    )


def _checkpoint_has_rllib_marker(path: Path) -> bool:
    return (path / "rllib_checkpoint.json").is_file()


def _latest_algorithm_checkpoint(artifacts_root: Path) -> Path:
    step_root = artifacts_root / "step_checkpoints"
    step_candidates = sorted(step_root.glob("steps_*"))
    for candidate in reversed(step_candidates):
        if _checkpoint_has_rllib_marker(candidate):
            return candidate

    checkpoint_root = artifacts_root / "checkpoints"
    iteration_candidates = sorted(checkpoint_root.glob("iteration_*"))
    for candidate in reversed(iteration_candidates):
        if _checkpoint_has_rllib_marker(candidate):
            return candidate

    raise FileNotFoundError(
        f"no RLlib checkpoint under {step_root} or {checkpoint_root}"
    )


def _resolve_resume_checkpoint(
    *,
    experiment_dir: Path,
    prior_run_id: str,
    artifacts_root: Path,
    results_dir: Path,
) -> Path:
    tune_summary_path = results_dir / "tune_summary.json"
    if tune_summary_path.is_file():
        tune_summary = json.loads(tune_summary_path.read_text())
        trials = tune_summary.get("trials") or []
        checkpoint_remote = str(trials[0].get("checkpoint") or "")
        if not checkpoint_remote:
            raise RuntimeError(
                f"prior trial checkpoint missing in {tune_summary_path}"
            )
        resume_from = _resume_checkpoint(
            experiment_dir=experiment_dir,
            prior_run_id=prior_run_id,
            checkpoint_remote=checkpoint_remote,
        )
    else:
        resume_from = _latest_algorithm_checkpoint(artifacts_root)

    if not resume_from.is_dir() or not _checkpoint_has_rllib_marker(resume_from):
        raise FileNotFoundError(
            f"restored checkpoint directory not found or invalid: {resume_from}"
        )
    return resume_from


def continue_prior_run(
    experiment_module: str,
    *,
    prior_run_id: str,
    target_agent_steps: int = CONTINUATION_TOTAL_ENV_STEPS,
    seed: int = 42,
    upload_artifacts: bool = True,
    hardware_profile: str = "cuda4090",
) -> None:
    experiment = load_experiment(experiment_module)
    results_dir = experiment.directory / "results" / prior_run_id
    if not results_dir.is_dir():
        raise FileNotFoundError(f"missing prior results directory: {results_dir}")

    manifest = _load_artifact_manifest(results_dir)
    artifacts_root = experiment.directory / "artifacts" / prior_run_id
    retrieve_manifest_artifacts(manifest, artifacts_root)

    prior_steps = _prior_agent_steps(results_dir)
    if target_agent_steps <= prior_steps:
        raise ValueError(
            f"target_agent_steps={target_agent_steps} must exceed "
            f"prior_agent_steps={prior_steps}"
        )
    resume_from = _resolve_resume_checkpoint(
        experiment_dir=experiment.directory,
        prior_run_id=prior_run_id,
        artifacts_root=artifacts_root,
        results_dir=results_dir,
    )
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
                "additional_agent_steps": target_agent_steps - prior_steps,
                "target_agent_steps": target_agent_steps,
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
            "--target-agent-steps",
            str(target_agent_steps),
            "--seed",
            str(seed),
            "--hardware-profile",
            hardware_profile,
        ],
        runtime_overrides={
            "prior_run_id": prior_run_id,
            "resume_from": str(resume_from),
            "prior_agent_steps": prior_steps,
            "target_agent_steps": target_agent_steps,
        },
        upload_artifacts=upload_artifacts,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a prior cycle-4 run's artifacts from B2 and continue "
            "training until a target lifetime env-step budget."
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
    parser.add_argument(
        "--target-agent-steps",
        type=int,
        default=CONTINUATION_TOTAL_ENV_STEPS,
        help=(
            "Stop once env_runners/num_env_steps_sampled_lifetime reaches this "
            "total budget."
        ),
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
        target_agent_steps=args.target_agent_steps,
        seed=args.seed,
        upload_artifacts=args.upload_artifacts,
        hardware_profile=args.hardware_profile,
    )


if __name__ == "__main__":
    main()
