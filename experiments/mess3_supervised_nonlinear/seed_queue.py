"""Run one or all nonlinear supervised arms across several seeds on one GPU box.

Designed for vast.ai: sequential seeds (and conditions), with an optional
compact-results push after each run so a mid-queue failure still lands completed
outputs on the ``results`` branch.

Example (on a provisioned box, already inside the activated ``.venv``):

  python -m experiments.mess3_supervised_nonlinear.seed_queue \\
    --condition two_layer_decoder --seeds 42 43 44

  python -m experiments.mess3_supervised_nonlinear.seed_queue \\
    --all-conditions --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

from harness.cli import execute_experiment, load_experiment, make_run_context

STUDY = "mess3_supervised_nonlinear"
ARMS = (
    "linear_decoder_control",
    "two_layer_decoder",
    "four_layer_decoder",
)
DEFAULT_SEEDS = (42, 43, 44)


def _instance_id() -> str | None:
    path = Path("/root/vast_instance_id")
    if path.is_file():
        return path.read_text().strip() or None
    return os.environ.get("VAST_INSTANCE_ID")


def _push_results(run_name: str) -> bool:
    """Push compact experiments/ results; no-op success when nothing staged."""

    try:
        from devops.vast.self_destruct import push_results
    except ImportError as error:
        print(f"[seed_queue] push unavailable: {error}", flush=True)
        return False
    return bool(
        push_results(
            branch=os.environ.get("VAST_RESULTS_BRANCH", "results"),
            run_name=run_name,
            instance_id=_instance_id(),
        )
    )


def _run_one(
    *,
    condition: str,
    seed: int,
    hardware_profile: str,
    upload_artifacts: bool,
    push_each: bool,
) -> int:
    module = f"experiments.{STUDY}.{condition}.experiment"
    run_id = f"{STUDY}-{condition}-seed{seed}"
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=False,
        hardware_profile=hardware_profile,
    )
    print(
        f"[seed_queue] start condition={condition} seed={seed} run_id={run_id}",
        flush=True,
    )
    started = time.time()
    try:
        execute_experiment(
            experiment,
            context,
            command=[
                "seed_queue",
                "--condition",
                condition,
                "--seed",
                str(seed),
            ],
            runtime_overrides={
                "hardware_profile": hardware_profile,
                "seed": seed,
                "smoke": False,
                "upload_artifacts": upload_artifacts,
            },
            upload_artifacts=upload_artifacts,
        )
    except Exception as error:  # noqa: BLE001 - keep the queue going
        print(f"[seed_queue] FAILED condition={condition} seed={seed}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1

    print(
        f"[seed_queue] done condition={condition} seed={seed} "
        f"elapsed_s={time.time() - started:.1f}",
        flush=True,
    )
    if push_each and not _push_results(run_id):
        print(f"[seed_queue] push failed condition={condition} seed={seed}", flush=True)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential multi-seed runner for mess3_supervised_nonlinear arm(s)."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--condition", choices=list(ARMS))
    group.add_argument(
        "--all-conditions",
        action="store_true",
        help="run every arm in order, each across the requested seeds",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--hardware-profile", default="cuda4090")
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--push-each",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="push compact results after each seed (default: on)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop the queue on the first run failure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = list(args.seeds)
    conditions = list(ARMS) if args.all_conditions else [args.condition]
    print(
        {
            "conditions": conditions,
            "seeds": seeds,
            "hardware_profile": args.hardware_profile,
            "upload_artifacts": args.upload_artifacts,
            "push_each": args.push_each,
        },
        flush=True,
    )
    codes: dict[str, dict[int, int]] = {}
    for condition in conditions:
        codes[condition] = {}
        for seed in seeds:
            codes[condition][seed] = _run_one(
                condition=condition,
                seed=seed,
                hardware_profile=args.hardware_profile,
                upload_artifacts=args.upload_artifacts,
                push_each=args.push_each,
            )
            if codes[condition][seed] != 0 and args.fail_fast:
                failed = {
                    arm: {s: c for s, c in seed_codes.items() if c != 0}
                    for arm, seed_codes in codes.items()
                    if any(c != 0 for c in seed_codes.values())
                }
                print({"seed_exit_codes": codes, "failed": failed}, flush=True)
                return 1
    failed = {
        condition: {seed: code for seed, code in seed_codes.items() if code != 0}
        for condition, seed_codes in codes.items()
        if any(code != 0 for code in seed_codes.values())
    }
    print({"seed_exit_codes": codes, "failed": failed}, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
