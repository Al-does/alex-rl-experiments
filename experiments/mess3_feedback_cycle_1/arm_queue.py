"""Run seeds 42–46 sequentially on one GPU and preserve each result."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

import experiments.mess3_feedback_cycle_1.aggregate as aggregate_module
from experiments.mess3_feedback_cycle_1.aggregate import write_summary
from experiments.mess3_feedback_cycle_1.shared import DEFAULT_SEEDS
from experiments.mess3_token_guess_cycle_2.arm_queue import _push_results
from harness.cli import execute_experiment, load_experiment, make_run_context


STUDY = "mess3_feedback_cycle_1"
MODULE = f"experiments.{STUDY}.ppo.experiment"


def _run_one(seed: int, *, upload_artifacts: bool, push_each: bool) -> int:
    run_id = f"{STUDY}-ppo-seed{seed}-2m"
    experiment = load_experiment(MODULE)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=False,
        hardware_profile="cuda4090",
    )
    print(f"[feedback_queue] start seed={seed} run_id={run_id}", flush=True)
    started = time.time()
    try:
        execute_experiment(
            experiment,
            context,
            command=["feedback_queue", "--seed", str(seed)],
            runtime_overrides={
                "hardware_profile": "cuda4090",
                "seed": seed,
                "smoke": False,
                "upload_artifacts": upload_artifacts,
            },
            upload_artifacts=upload_artifacts,
        )
    except Exception as error:  # noqa: BLE001 - preserve completed seed outputs
        print(f"[feedback_queue] FAILED seed={seed}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1
    print(
        f"[feedback_queue] done seed={seed} elapsed_s={time.time() - started:.1f}",
        flush=True,
    )
    if push_each and not _push_results(run_id):
        print(f"[feedback_queue] result push failed seed={seed}", flush=True)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential five-seed runner for MESS3 feedback cycle 1."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--push-each",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    codes: dict[int, int] = {}
    for seed in args.seeds:
        code = _run_one(
            int(seed),
            upload_artifacts=bool(args.upload_artifacts),
            push_each=bool(args.push_each),
        )
        codes[int(seed)] = code
        if code and args.fail_fast:
            break

    failed = [seed for seed, code in codes.items() if code]
    expected_complete = list(args.seeds) == list(DEFAULT_SEEDS) and not failed
    if expected_complete:
        summary_dir = write_summary(
            Path(aggregate_module.__file__).resolve().parent / "ppo" / "results"
        )
        print(f"[feedback_queue] wrote aggregate {summary_dir}", flush=True)
        if args.push_each and not _push_results(
            "mess3-feedback-c1-five-seed-summary"
        ):
            failed.append(-1)
    print({"seed_exit_codes": codes, "failed": failed}, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
