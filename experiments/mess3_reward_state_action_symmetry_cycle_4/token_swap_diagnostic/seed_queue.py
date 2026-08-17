"""Recover final checkpoints and run token-swap diagnostics sequentially.

Example:
  python -m experiments.mess3_reward_state_action_symmetry_cycle_4.token_swap_diagnostic.seed_queue \
    v2:42 v2:43 v2:44 v2:45 v2:46
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import re
import time
from typing import Any

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    _instance_id,
    _parse_job,
    _push_results,
    recover_bundle,
)
from harness.cli import execute_experiment, load_experiment, make_run_context

VARIANT = 2


def run_job(*, cycle: int, variant: int, seed: int, push_each: bool) -> int:
    package = f"mess3_reward_state_action_symmetry_cycle_{cycle}"
    module = f"experiments.{package}.token_swap_diagnostic.experiment"
    run_id = f"mess3-rsa-c{cycle}-token-swap-v{variant}-seed{seed}"
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        hardware_profile="cuda4090",
    )
    bundle = context.artifacts_dir / "source_checkpoint_bundle"
    try:
        recover_bundle(
            cycle=cycle,
            variant=variant,
            seed=seed,
            destination=bundle,
        )
        context = make_run_context(
            experiment,
            seed=seed,
            run_id=run_id,
            resume_from=bundle,
            results_dir=context.results_dir,
            artifacts_dir=context.artifacts_dir,
            hardware_profile="cuda4090",
        )
        started = time.time()
        execute_experiment(
            experiment,
            context,
            command=["seed_queue", f"v{variant}:{seed}"],
            runtime_overrides={
                "hardware_profile": "cuda4090",
                "seed": seed,
                "resume_from": str(bundle),
                "upload_artifacts": False,
            },
            upload_artifacts=False,
        )
        print(
            f"[token_swap_queue] completed {run_id} in {time.time() - started:.1f}s",
            flush=True,
        )
    except Exception as error:
        print(f"[token_swap_queue] FAILED {run_id}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1
    if push_each and not _push_results(run_id):
        return 2
    return 0


def main(argv: Sequence[str] | None = None, *, cycle: int = 4) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", type=_parse_job)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--push-each", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    codes: dict[str, int] = {}
    for variant, seed in args.jobs:
        if variant != VARIANT:
            raise SystemExit(f"token-swap diagnostic only supports variant {VARIANT}")
        label = f"v{variant}:{seed}"
        codes[label] = run_job(
            cycle=cycle,
            variant=variant,
            seed=seed,
            push_each=args.push_each,
        )
        if codes[label] and args.fail_fast:
            break
    print({"job_exit_codes": codes}, flush=True)
    return 1 if any(codes.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
