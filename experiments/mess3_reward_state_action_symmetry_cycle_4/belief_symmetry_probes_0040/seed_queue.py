"""Recover longitudinal checkpoints and run campaign-0040 belief probes.

Example:
  python -m experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes_0040.seed_queue \
    v1:42 v1:43 v1:44
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes_0040.analysis import (
    PROBE_ITERATIONS,
    _checkpoint_name_for_iteration,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    HISTORICAL_STUDIES,
    SOURCE_RUN_PATTERNS,
    _candidate_bases,
    _download_atomic,
    _final_checkpoint_name,
    _objects,
    _parse_job,
    _push_results,
    _select_source_base,
    _validate_checkpoint,
    _walk_strings,
)
from harness.cli import execute_experiment, load_experiment, make_run_context

CHECKPOINT_NAME = re.compile(r"checkpoint_\d+")
CAMPAIGN_SUFFIX = "0040"


def _required_tune_checkpoints(
    tune_summary: dict[str, Any],
) -> dict[str, str]:
    """Return bundle member labels mapped to tune checkpoint directory names."""
    names = {
        match.group(0)
        for value in _walk_strings(tune_summary)
        for match in CHECKPOINT_NAME.finditer(value)
    }
    if not names:
        raise ValueError("tune_summary.json contains no checkpoint name")
    final_name = _final_checkpoint_name(tune_summary)
    required = {"iter_22": final_name}
    for iteration in PROBE_ITERATIONS:
        if iteration == 22:
            continue
        expected = _checkpoint_name_for_iteration(iteration)
        if expected not in names:
            raise FileNotFoundError(
                f"tune summary lacks required checkpoint {expected} for iteration {iteration}"
            )
        required[f"iter_{iteration}"] = expected
    return required


def recover_bundle(
    *,
    cycle: int,
    variant: int,
    seed: int,
    destination: Path,
) -> Path:
    """Download initial plus iterations 2, 8, and 22 checkpoints from B2."""
    from harness.storage.b2 import B2StorageConfig

    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 credentials are required to recover source checkpoints")
    client = config.s3_client()
    study = HISTORICAL_STUDIES[cycle]
    source_run_id = SOURCE_RUN_PATTERNS[cycle].format(variant=variant, seed=seed)
    base, tune_key = _select_source_base(
        client,
        config.bucket,
        _candidate_bases(
            configured_prefix=config.prefix,
            study=study,
            variant=variant,
            source_run_id=source_run_id,
        ),
    )
    tune_response = client.get_object(Bucket=config.bucket, Key=tune_key)
    tune_body = tune_response["Body"]
    try:
        tune_summary = json.loads(tune_body.read())
    finally:
        tune_body.close()
    tune_checkpoints = _required_tune_checkpoints(tune_summary)
    all_keys = _objects(client, config.bucket, f"{base}/")
    selected: list[tuple[str, str, str]] = []
    for key in all_keys:
        if "/initial_checkpoint/" in key:
            relative = key.split("/initial_checkpoint/", 1)[1]
            selected.append((key, "initial_checkpoint", relative))
            continue
        for member, checkpoint_name in tune_checkpoints.items():
            if f"/{checkpoint_name}/" in key and "/compact-results/" not in key:
                relative = key.split(f"/{checkpoint_name}/", 1)[1]
                selected.append((key, member, relative))
                break
    expected_members = {"initial_checkpoint", *tune_checkpoints.keys()}
    recovered_members = {member for _, member, _ in selected}
    missing = sorted(expected_members - recovered_members)
    if missing:
        raise FileNotFoundError(
            f"no checkpoint objects found for bundle members: {missing}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for key, member, relative in selected:
            _download_atomic(client, config.bucket, key, staging / member / relative)
        _validate_checkpoint(staging / "initial_checkpoint")
        for member in tune_checkpoints:
            _validate_checkpoint(staging / member)
        (staging / "source_provenance.json").write_text(
            json.dumps(
                {
                    "source_run_id": source_run_id,
                    "historical_study": study,
                    "cycle": cycle,
                    "variant": variant,
                    "seed": seed,
                    "b2_base": base,
                    "probe_iterations": list(PROBE_ITERATIONS),
                    "tune_checkpoints": tune_checkpoints,
                },
                indent=2,
            )
            + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def run_job(*, cycle: int, variant: int, seed: int, push_each: bool) -> int:
    package = f"mess3_reward_state_action_symmetry_cycle_{cycle}"
    module = (
        f"experiments.{package}.belief_symmetry_probes_0040.variant_{variant}.experiment"
    )
    run_id = (
        f"mess3-rsa-c{cycle}-belief-symmetry-probe-{CAMPAIGN_SUFFIX}-"
        f"v{variant}-seed{seed}"
    )
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
        print(f"[seed_queue] completed {run_id} in {time.time() - started:.1f}s", flush=True)
    except Exception as error:
        print(f"[seed_queue] FAILED {run_id}: {error}", flush=True)
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
        label = f"v{variant}:{seed}"
        codes[label] = run_job(
            cycle=cycle, variant=variant, seed=seed, push_each=args.push_each
        )
        if codes[label] and args.fail_fast:
            break
    print({"job_exit_codes": codes}, flush=True)
    return 1 if any(codes.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
