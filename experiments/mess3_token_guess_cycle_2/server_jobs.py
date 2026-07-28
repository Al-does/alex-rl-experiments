"""Launch token-guess cycle-2 full runs on RunPod Flash (vast.ai fallback).

Flash keeps the endpoint after each job (workers.min=0), so sequential jobs
reuse the same deployment without re-renting. Parallel mode raises
``--max-workers`` so each concurrent seed can land on its own GPU worker.

Results are durable via ``--upload-artifacts`` (B2) and ``--self-destruct``
(Git ``results`` branch publication). Example:

  uv run --directory /rl-harness --group flash \\
    python experiments/mess3_token_guess_cycle_2/server_jobs.py --dry-run

  # then the same command with --yes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

CONDITIONS = (
    "a2c",
    "ppo",
    "predictive_loss",
    "decoupled_kelly",
    "iqn",
)
DEFAULT_SEEDS = (42, 43, 44)
STUDY = "mess3_token_guess_cycle_2"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARNESS = Path("/rl-harness")
DEFAULT_APP = "rlh-flash-mess3-tg-c2"


@dataclass(frozen=True, slots=True)
class JobSpec:
    condition: str
    seed: int

    @property
    def module(self) -> str:
        return f"experiments.{STUDY}.{self.condition}.experiment"

    @property
    def run_name(self) -> str:
        return f"{STUDY}-{self.condition}-seed{self.seed}"


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()


def _flash_cmd(harness: Path, *args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--directory",
        str(harness),
        "--group",
        "flash",
        "python",
        "-m",
        "devops.flash.provision",
        *args,
    ]


def _vast_cmd(harness: Path, *args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--directory",
        str(harness),
        "--group",
        "devops",
        "python",
        "-m",
        "devops.vast.provision",
        *args,
    ]


def _parse_endpoint_id(deploy_output: str) -> str | None:
    for line in deploy_output.splitlines():
        if "verified endpoint " in line:
            token = line.split("verified endpoint ", 1)[1].split(":", 1)[0]
            return token.strip()
    return None


def deploy_flash(
    *,
    harness: Path,
    app: str,
    max_workers: int,
    dry_run: bool,
    yes: bool,
) -> str | None:
    argv = _flash_cmd(
        harness,
        "deploy",
        "--app",
        app,
        "--environment",
        "production",
        "--max-workers",
        str(max_workers),
    )
    if dry_run:
        argv.append("--dry-run")
    elif yes:
        argv.append("--yes")
    else:
        raise ValueError("live deploy requires --yes (or use --dry-run)")
    print("+ " + " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False, text=True, capture_output=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Flash deploy failed with code {completed.returncode}"
        )
    if dry_run:
        return None
    endpoint_id = _parse_endpoint_id(completed.stdout)
    if not endpoint_id:
        raise RuntimeError("Flash deploy succeeded but endpoint id was not parsed")
    return endpoint_id


def build_run_command(job: JobSpec) -> str:
    # Use --hardware so Flash/Serverless preflight and the experiment CLI agree.
    return (
        f"rl-harness {job.module} "
        f"--seed {job.seed} "
        f"--hardware cuda4090 "
        f"--upload-artifacts "
        f"--run-id {job.run_name}"
    )


def flash_up(
    *,
    harness: Path,
    endpoint_id: str,
    job: JobSpec,
    experiment_ref: str,
    library_ref: str,
    max_age: float,
    queue_timeout: float,
    max_price: float,
    max_estimated_cost: float,
    dry_run: bool,
    yes: bool,
) -> int:
    argv = _flash_cmd(
        harness,
        "up",
        "--endpoint-id",
        endpoint_id,
        "--experiment-ref",
        experiment_ref,
        "--library-ref",
        library_ref,
        "--run-name",
        job.run_name,
        "--run",
        build_run_command(job),
        "--max-age",
        str(max_age),
        "--queue-timeout",
        str(queue_timeout),
        "--max-price",
        str(max_price),
        "--max-estimated-cost",
        str(max_estimated_cost),
        "--forward-b2",
        "--self-destruct",
    )
    if dry_run:
        argv.append("--dry-run")
    elif yes:
        argv.append("--yes")
    else:
        raise ValueError("live up requires --yes (or use --dry-run)")
    print("+ " + " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False, text=True)
    return int(completed.returncode)


def vast_up(
    *,
    harness: Path,
    job: JobSpec,
    experiment_ref: str,
    library_ref: str,
    max_age: float,
    max_price: float,
    dry_run: bool,
    yes: bool,
) -> int:
    argv = _vast_cmd(
        harness,
        "up",
        "-n",
        "1",
        "--commit",
        experiment_ref,
        "--library-commit",
        library_ref,
        "--run",
        build_run_command(job),
        "--run-name",
        job.run_name,
        "--max-age",
        str(max_age),
        "--max-price",
        str(max_price),
        "--forward-b2",
        "--self-destruct",
        "--no-open",
    )
    if dry_run:
        argv.append("--dry-run")
    elif yes:
        argv.append("--yes")
    else:
        raise ValueError("live vast up requires --yes (or use --dry-run)")
    print("+ " + " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False, text=True)
    return int(completed.returncode)


def plan_jobs(
    conditions: Sequence[str],
    seeds: Sequence[int],
) -> list[JobSpec]:
    return [
        JobSpec(condition=condition, seed=seed)
        for condition in conditions
        for seed in seeds
    ]


def run_flash_queue(
    *,
    harness: Path,
    endpoint_id: str,
    jobs: Sequence[JobSpec],
    experiment_ref: str,
    library_ref: str,
    max_workers: int,
    max_age: float,
    queue_timeout: float,
    max_price: float,
    max_estimated_cost: float,
    dry_run: bool,
    yes: bool,
) -> dict[str, Any]:
    """Run at most ``max_workers`` Flash ups concurrently; reuse one endpoint."""

    results: dict[str, int] = {}
    if dry_run and endpoint_id in {"", "dry-run-endpoint"}:
        for job in jobs:
            print(
                json.dumps(
                    {
                        "planned_flash_job": job.run_name,
                        "run": build_run_command(job),
                        "experiment_ref": experiment_ref,
                        "library_ref": library_ref,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            results[job.run_name] = 0
        return {"backend": "flash", "results": results}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                flash_up,
                harness=harness,
                endpoint_id=endpoint_id,
                job=job,
                experiment_ref=experiment_ref,
                library_ref=library_ref,
                max_age=max_age,
                queue_timeout=queue_timeout,
                max_price=max_price,
                max_estimated_cost=max_estimated_cost,
                dry_run=dry_run,
                yes=yes,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results[job.run_name] = int(future.result())
            except Exception as error:  # noqa: BLE001 - surface per-job failure
                print(f"{job.run_name} raised: {error}", flush=True)
                results[job.run_name] = 2
    return {"backend": "flash", "results": results}


def run_vast_queue(
    *,
    harness: Path,
    jobs: Sequence[JobSpec],
    experiment_ref: str,
    library_ref: str,
    max_workers: int,
    max_age: float,
    max_price: float,
    dry_run: bool,
    yes: bool,
) -> dict[str, Any]:
    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                vast_up,
                harness=harness,
                job=job,
                experiment_ref=experiment_ref,
                library_ref=library_ref,
                max_age=max_age,
                max_price=max_price,
                dry_run=dry_run,
                yes=yes,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results[job.run_name] = int(future.result())
            except Exception as error:  # noqa: BLE001
                print(f"{job.run_name} raised: {error}", flush=True)
                results[job.run_name] = 2
    return {"backend": "vast", "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch mess3_token_guess_cycle_2 full training jobs on RunPod "
            "Flash with vast.ai fallback. Endpoint is reused across jobs."
        )
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--mode",
        choices=("parallel-seeds", "sequential"),
        default="parallel-seeds",
        help=(
            "parallel-seeds: max-workers=len(seeds) so seeds can use different "
            "GPUs; sequential: max-workers=1 and jobs share one machine over time"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--app", default=DEFAULT_APP)
    parser.add_argument("--endpoint-id", default=None)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--experiment-ref", default=None)
    parser.add_argument("--library-ref", default=None)
    parser.add_argument("--max-age", type=float, default=12.0)
    parser.add_argument("--queue-timeout", type=float, default=30.0)
    parser.add_argument("--max-price", type=float, default=1.25)
    parser.add_argument("--max-estimated-cost", type=float, default=20.0)
    parser.add_argument(
        "--backend",
        choices=("flash", "vast", "flash-then-vast"),
        default="flash-then-vast",
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=EXPERIMENT_ROOT
        / "experiments"
        / STUDY
        / "artifacts"
        / "server_jobs_state.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.yes:
        raise SystemExit("pass only one of --dry-run / --yes")
    if not args.dry_run and not args.yes:
        raise SystemExit("pass --dry-run or --yes")

    harness = args.harness.expanduser().resolve()
    if not harness.is_dir():
        raise SystemExit(f"harness checkout missing: {harness}")

    jobs = plan_jobs(args.conditions, args.seeds)
    if args.mode == "sequential":
        max_workers = 1
    else:
        max_workers = args.max_workers or max(1, len(args.seeds))
    if max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")

    experiment_ref = args.experiment_ref or _git_sha(EXPERIMENT_ROOT)
    library_ref = args.library_ref or _git_sha(harness)
    print(
        json.dumps(
            {
                "jobs": [job.run_name for job in jobs],
                "mode": args.mode,
                "max_workers": max_workers,
                "experiment_ref": experiment_ref,
                "library_ref": library_ref,
                "backend": args.backend,
            },
            indent=2,
        ),
        flush=True,
    )

    state: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_ref": experiment_ref,
        "library_ref": library_ref,
        "mode": args.mode,
        "max_workers": max_workers,
        "jobs": [job.run_name for job in jobs],
    }

    prefer_flash = args.backend in {"flash", "flash-then-vast"}
    prefer_vast = args.backend in {"vast", "flash-then-vast"}
    summary: dict[str, Any] | None = None

    if prefer_flash:
        try:
            endpoint_id = args.endpoint_id
            if endpoint_id is None:
                endpoint_id = deploy_flash(
                    harness=harness,
                    app=args.app,
                    max_workers=max_workers,
                    dry_run=args.dry_run,
                    yes=args.yes,
                )
            state["endpoint_id"] = endpoint_id
            if not args.dry_run and not endpoint_id:
                raise RuntimeError("missing Flash endpoint id")
            summary = run_flash_queue(
                harness=harness,
                endpoint_id=endpoint_id or "dry-run-endpoint",
                jobs=jobs,
                experiment_ref=experiment_ref,
                library_ref=library_ref,
                max_workers=max_workers,
                max_age=args.max_age,
                queue_timeout=args.queue_timeout,
                max_price=args.max_price,
                max_estimated_cost=args.max_estimated_cost,
                dry_run=args.dry_run,
                yes=args.yes,
            )
        except Exception as error:  # noqa: BLE001
            print(f"Flash path failed: {error}", flush=True)
            state["flash_error"] = str(error)
            if not prefer_vast or args.backend == "flash":
                raise
            summary = None

    if summary is None and prefer_vast:
        print("Falling back to vast.ai", flush=True)
        summary = run_vast_queue(
            harness=harness,
            jobs=jobs,
            experiment_ref=experiment_ref,
            library_ref=library_ref,
            max_workers=max_workers,
            max_age=args.max_age,
            max_price=args.max_price,
            dry_run=args.dry_run,
            yes=args.yes,
        )

    assert summary is not None
    state["summary"] = summary
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.state_out}", flush=True)
    failed = [name for name, code in summary["results"].items() if code != 0]
    if failed:
        print(f"failed jobs: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
