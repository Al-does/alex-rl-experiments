"""Run one token-guess cycle-2 arm across many seeds on a single GPU box.

Designed for vast.ai: sequential seeds, truncated at the third checkpoint
(~0.66M steps), optional compact-results push after each seed so a mid-queue
failure still lands completed runs on the ``results`` branch.

Example (on a provisioned box, already in the activated .venv):

  python experiments/mess3_token_guess_cycle_2/arm_queue.py \\
    --condition decoupled_kelly --seeds 42 43 44 ... 56
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

from experiments.mess3_token_guess_cycle_2.shared import THIRD_CHECKPOINT_ENV_STEPS
from harness.cli import execute_experiment, load_experiment, make_run_context

STUDY = "mess3_token_guess_cycle_2"
ARMS = ("ppo", "predictive_loss", "decoupled_kelly")
DEFAULT_SEEDS = tuple(range(42, 57))  # 15 seeds: 42..56


def _instance_id() -> str | None:
    path = Path("/root/vast_instance_id")
    if path.is_file():
        text = path.read_text().strip()
        return text or None
    return os.environ.get("VAST_INSTANCE_ID")


def _push_results(run_name: str) -> bool:
    """Push compact experiments/ results; no-op success when nothing staged."""

    try:
        from devops.vast.self_destruct import push_results
    except ImportError as error:
        print(f"[arm_queue] push unavailable: {error}", flush=True)
        return False
    branch = os.environ.get("VAST_RESULTS_BRANCH", "results")
    return bool(
        push_results(
            branch=branch,
            run_name=run_name,
            instance_id=_instance_id(),
        )
    )


def _run_one(
    *,
    condition: str,
    seed: int,
    max_env_steps: int,
    upload_artifacts: bool,
    push_each: bool,
) -> int:
    module = f"experiments.{STUDY}.{condition}.experiment"
    run_id = f"{STUDY}-{condition}-seed{seed}-0p66m"
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=False,
        hardware_profile="cuda4090",
    )
    os.environ["MESS3_TG_C2_MAX_ENV_STEPS"] = str(max_env_steps)
    print(
        f"[arm_queue] start condition={condition} seed={seed} "
        f"run_id={run_id} max_env_steps={max_env_steps}",
        flush=True,
    )
    started = time.time()
    try:
        execute_experiment(
            experiment,
            context,
            command=[
                "arm_queue",
                "--condition",
                condition,
                "--seed",
                str(seed),
                "--max-env-steps",
                str(max_env_steps),
            ],
            runtime_overrides={
                "hardware_profile": "cuda4090",
                "seed": seed,
                "smoke": False,
                "max_env_steps": max_env_steps,
                "upload_artifacts": upload_artifacts,
            },
            upload_artifacts=upload_artifacts,
        )
    except Exception as error:  # noqa: BLE001 - keep queue going / surface failure
        print(f"[arm_queue] FAILED seed={seed}: {error}", flush=True)
        if push_each:
            # Still try to salvage any compact outputs written before failure.
            _push_results(f"{run_id}-failed")
        return 1

    elapsed = time.time() - started
    print(f"[arm_queue] done seed={seed} elapsed_s={elapsed:.1f}", flush=True)
    if push_each:
        ok = _push_results(run_id)
        print(f"[arm_queue] push_each seed={seed} ok={ok}", flush=True)
        if not ok:
            return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential multi-seed runner for one mess3_token_guess_cycle_2 arm, "
            "truncated at the third checkpoint (~0.66M steps)."
        )
    )
    parser.add_argument("--condition", required=True, choices=list(ARMS))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=THIRD_CHECKPOINT_ENV_STEPS,
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
        help="push compact results after each seed (default: on)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop the queue on the first seed failure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = list(args.seeds)
    print(
        {
            "condition": args.condition,
            "seeds": seeds,
            "max_env_steps": args.max_env_steps,
            "upload_artifacts": args.upload_artifacts,
            "push_each": args.push_each,
        },
        flush=True,
    )
    codes: dict[int, int] = {}
    for seed in seeds:
        code = _run_one(
            condition=args.condition,
            seed=seed,
            max_env_steps=args.max_env_steps,
            upload_artifacts=args.upload_artifacts,
            push_each=args.push_each,
        )
        codes[seed] = code
        if code != 0 and args.fail_fast:
            break
    failed = [seed for seed, code in codes.items() if code != 0]
    print({"seed_exit_codes": codes, "failed": failed}, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
