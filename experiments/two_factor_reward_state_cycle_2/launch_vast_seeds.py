"""Rent five on-demand vast.ai boxes — one seed each, four arms per box.

Dry-run first, then live:

  uv run python -m experiments.two_factor_reward_state_cycle_2.launch_vast_seeds --dry-run
  uv run python -m experiments.two_factor_reward_state_cycle_2.launch_vast_seeds --yes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

STUDY = "two_factor_reward_state_cycle_2"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_MAX_AGE_HOURS = 30.0
TOTAL_ENV_STEPS = 5_000_000
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_harness(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for candidate in (
        EXPERIMENT_ROOT.parent / "rl-harness",
        Path("/rl-harness"),
        Path("/agent/repos/RL-Harness"),
        Path("/agent/repos/rl-harness"),
    ):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("could not locate rl-harness checkout")


def _git_ref(repo: Path) -> str:
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if branch != "HEAD":
        return branch
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()


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


def build_run_command(seed: int) -> str:
    return (
        "python -m experiments.two_factor_reward_state_cycle_2.seed_queue "
        f"--seed {seed} --hardware-profile cuda4090"
    )


def provision_seed(
    *,
    harness: Path,
    experiment_repo: Path,
    seed: int,
    experiment_ref: str,
    library_ref: str,
    max_age: float,
    max_price: float,
    dry_run: bool,
    yes: bool,
    exclude_machines: Sequence[int] = (),
) -> int:
    run_name = f"two-factor-cycle2-seed{seed}-5m-x4"
    argv = _vast_cmd(
        harness,
        "up",
        "-n",
        "1",
        "--mode",
        "ondemand",
        "--experiment-repo",
        str(experiment_repo),
        "--branch",
        experiment_ref,
        "--library-commit",
        library_ref,
        "--run",
        build_run_command(seed),
        "--run-name",
        run_name,
        "--max-age",
        str(max_age),
        "--max-price",
        str(max_price),
        "--forward-b2",
        "--self-destruct",
        "--regions",
        "US,CA",
        "--no-open",
    )
    if exclude_machines:
        argv.append("--exclude-machine")
        argv.extend(str(int(machine_id)) for machine_id in exclude_machines)
    if dry_run:
        argv.append("--dry-run")
    elif yes:
        argv.append("--yes")
    else:
        raise ValueError("pass --dry-run or --yes")
    print("+ " + " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False, text=True)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch five on-demand vast boxes for cycle-2 "
            "(four 5M-step arms per seed)."
        )
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--experiment-repo", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--experiment-ref", default=None)
    parser.add_argument("--library-ref", default=None)
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument("--max-price", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--exclude-machine",
        type=int,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=EXPERIMENT_ROOT
        / "experiments"
        / STUDY
        / "artifacts"
        / "vast_launch_state.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.yes:
        raise SystemExit("pass only one of --dry-run / --yes")
    if not args.dry_run and not args.yes:
        raise SystemExit("pass --dry-run or --yes")

    harness = _resolve_harness(args.harness)
    experiment_repo = args.experiment_repo.expanduser().resolve()
    experiment_ref = args.experiment_ref or _git_ref(experiment_repo)
    library_ref = args.library_ref or _git_sha(harness)
    seeds = list(args.seeds)
    plan = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": seeds,
        "arms_per_seed": [
            "SAC reward_both",
            "SAC reward_factor_1",
            "PPO reward_both",
            "PPO reward_factor_1",
        ],
        "total_env_steps_per_arm": TOTAL_ENV_STEPS,
        "experiment_ref": experiment_ref,
        "library_ref": library_ref,
        "max_age_hours": args.max_age,
        "exclude_machines": list(args.exclude_machine),
        "run_commands": {
            str(seed): build_run_command(seed) for seed in seeds
        },
    }
    print(json.dumps(plan, indent=2), flush=True)

    results: dict[str, int] = {}
    for seed in seeds:
        code = provision_seed(
            harness=harness,
            experiment_repo=experiment_repo,
            seed=seed,
            experiment_ref=experiment_ref,
            library_ref=library_ref,
            max_age=args.max_age,
            max_price=args.max_price,
            dry_run=args.dry_run,
            yes=args.yes,
            exclude_machines=args.exclude_machine,
        )
        results[str(seed)] = code
        if code != 0:
            print(f"[launch] seed={seed} provision exited {code}", flush=True)

    plan["provision_exit_codes"] = results
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.state_out}", flush=True)
    return 1 if any(code != 0 for code in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
