"""Restore a completed Pusher-B PPO checkpoint from B2 and continue training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from devops.serverless.retrieve import retrieve_manifest_artifacts
from harness.cli import execute_experiment, load_experiment, make_run_context
from harness.storage.b2 import B2StorageConfig

STUDY_DIR = Path(__file__).resolve().parent


def _prior_checkpoint_remote(prior_leaf: str, prior_run_id: str) -> str:
    tune_summary_path = (
        STUDY_DIR / prior_leaf / "results" / prior_run_id / "tune_summary.json"
    )
    if not tune_summary_path.is_file():
        raise FileNotFoundError(f"missing tune summary: {tune_summary_path}")
    trials = json.loads(tune_summary_path.read_text()).get("trials") or []
    if len(trials) != 1:
        raise RuntimeError(
            f"expected one prior trial in {tune_summary_path}, found {len(trials)}"
        )
    checkpoint = str(trials[0].get("checkpoint") or "")
    if not checkpoint:
        raise RuntimeError(f"prior trial checkpoint missing in {tune_summary_path}")
    return checkpoint


def fetch_prior_checkpoint(
    *,
    prior_leaf: str,
    prior_run_id: str,
    destination: Path,
) -> Path:
    """Download only the final Tune checkpoint of a prior run from B2."""
    manifest_summary = json.loads(
        (
            STUDY_DIR / prior_leaf / "results" / prior_run_id / "run_manifest.json"
        ).read_text()
    )["remote_artifacts"]
    if manifest_summary.get("status") != "completed":
        raise RuntimeError("prior run's B2 upload did not complete")
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 credentials are not configured")
    client = config.s3_client()
    body = client.get_object(
        Bucket=manifest_summary["bucket"],
        Key=manifest_summary["canonical_manifest_key"],
    )["Body"].read()
    manifest = json.loads(body)

    marker = f"/artifacts/{prior_run_id}/"
    remote = _prior_checkpoint_remote(prior_leaf, prior_run_id)
    if marker not in remote:
        raise ValueError("tune_summary checkpoint path does not match run id")
    relative = remote.split(marker, 1)[1].rstrip("/")
    files = [
        row
        for row in manifest["files"]
        if str(row.get("relative_path", "")).startswith(relative + "/")
    ]
    if not files:
        raise RuntimeError(f"no checkpoint files under {relative!r} in B2 manifest")
    retrieve_manifest_artifacts(
        {**manifest, "files": files},
        destination,
        client=client,
        config=config,
    )
    resume_from = destination / relative
    if not (resume_from / "rllib_checkpoint.json").is_file():
        raise FileNotFoundError(f"restored checkpoint incomplete: {resume_from}")
    return resume_from


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_module")
    parser.add_argument("--prior-leaf", required=True)
    parser.add_argument("--prior-run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="local checkpoint directory; skips the B2 download",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hardware-profile", "--hardware", default="auto")
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args(argv)

    experiment = load_experiment(args.experiment_module)
    resume_from = (
        args.resume_from.resolve()
        if args.resume_from is not None
        else fetch_prior_checkpoint(
            prior_leaf=args.prior_leaf,
            prior_run_id=args.prior_run_id,
            destination=(
                experiment.directory / "artifacts" / "prior" / args.prior_run_id
            ),
        )
    )
    context = make_run_context(
        experiment,
        seed=args.seed,
        smoke=args.smoke,
        resume_from=resume_from,
        hardware_profile=args.hardware_profile,
    )
    execute_experiment(
        experiment,
        context,
        command=[
            "python",
            "-m",
            "experiments.pusher_b.continue_ppo",
            args.experiment_module,
            "--prior-leaf",
            args.prior_leaf,
            "--prior-run-id",
            args.prior_run_id,
            "--seed",
            str(args.seed),
        ],
        runtime_overrides={
            "prior_leaf": args.prior_leaf,
            "prior_run_id": args.prior_run_id,
            "resume_from": str(resume_from),
            "settings_overrides": os.environ.get("PUSHER_B_CONTINUE_OVERRIDES"),
        },
        upload_artifacts=args.upload_artifacts,
    )


if __name__ == "__main__":
    main()
