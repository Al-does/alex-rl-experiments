"""Post-hoc held-out belief probes for Cycle-6 continuation checkpoints.

The continuation runs saved step-checkpoints every 25M env steps and uploaded
them to B2 at save time, but ran no probes. This driver downloads each
checkpoint from B2 (cached under ``variant_N/artifacts/<run_id>/``) and applies
the same ``probe_checkpoint`` protocol the battery used at training time,
writing ``checkpoint_probes/steps_*/probe_metrics.json``,
``checkpoint_probe_curve.json``, and ``belief_simplex_final.png`` into the
run's results directory, then augments ``condition_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
from collections.abc import Sequence
from pathlib import Path

from experiments.mess3_reward_state_action_symmetry_cycle_6.analysis import (
    plot_probe,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.continue_queue import (
    STUDY,
    STUDY_DIR,
    _checkpoint_valid,
    _load_json,
    _storage_config,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    _probe_at,
)
from harness.cli import load_experiment, make_run_context
from harness.hardware import configure_hardware


def _continued_run_id(variant: int, seed: int, target_steps: int) -> str:
    return (
        f"{STUDY}-variant{variant}-continued-seed{seed}"
        f"-{target_steps // 1_000_000}m"
    )


def _results_dir(variant: int, run_id: str) -> Path:
    path = STUDY_DIR / f"variant_{variant}" / "results" / run_id
    if not path.is_dir():
        raise FileNotFoundError(f"continued run results not found: {path}")
    return path


def _remote_prefix(variant: int, run_id: str) -> tuple[dict, str]:
    manifest = _load_json(_results_dir(variant, run_id) / "run_manifest.json")
    remote = manifest["remote_artifacts"]
    base_uri = str(remote["base_uri"])
    bucket_prefix = f"s3://{remote['bucket']}/"
    if not base_uri.startswith(bucket_prefix):
        raise ValueError(f"unexpected remote base_uri: {base_uri}")
    return remote, base_uri[len(bucket_prefix):].rstrip("/") + "/"


def _planned_checkpoints(variant: int, run_id: str) -> list[dict]:
    """step_checkpoints entries + the final iteration checkpoint, deduped."""
    results = _results_dir(variant, run_id)
    planned: list[dict] = []
    seen_steps: set[int] = set()
    uploads = results / "checkpoint_uploads.jsonl"
    for line in uploads.read_text().splitlines():
        entry = json.loads(line)
        if not entry.get("uploaded"):
            continue
        steps = int(entry["agent_steps"])
        if steps in seen_steps:
            continue
        seen_steps.add(steps)
        planned.append(
            {
                "name": entry["checkpoint_name"],
                "remote_dir": "step_checkpoints",
                "agent_steps": steps,
            }
        )
    summary = _load_json(results / "condition_summary.json")
    final_steps = int(summary["final_agent_steps"])
    if final_steps not in seen_steps:
        planned.append(
            {
                "name": None,  # resolved by listing checkpoints/ on B2
                "remote_dir": "checkpoints",
                "agent_steps": final_steps,
            }
        )
    return planned


def _download_checkpoint(
    client,
    *,
    bucket: str,
    prefix: str,
    remote_dir: str,
    name: str | None,
    destination: Path,
) -> Path:
    if name is not None and _checkpoint_valid(destination / name):
        return destination / name
    paginator = client.get_paginator("list_objects_v2")
    objects = [
        item["Key"]
        for page in paginator.paginate(
            Bucket=bucket, Prefix=f"{prefix}{remote_dir}/"
        )
        for item in page.get("Contents", [])
    ]
    if name is None:
        finals = sorted(
            {
                key.split(f"{remote_dir}/", 1)[1].split("/", 1)[0]
                for key in objects
            }
        )
        if len(finals) != 1:
            raise RuntimeError(
                f"expected one final checkpoint under {prefix}{remote_dir}/, "
                f"found {finals}"
            )
        name = finals[0]
    marker = f"{prefix}{remote_dir}/{name}/"
    members = [key for key in objects if key.startswith(marker)]
    if not members:
        raise FileNotFoundError(f"remote checkpoint not found: {marker}")
    target = destination / name
    if _checkpoint_valid(target):
        return target
    for key in members:
        relative = Path(key.split(marker, 1)[1])
        out = target / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(out))
    if not _checkpoint_valid(target):
        raise RuntimeError(f"downloaded checkpoint is incomplete: {target}")
    _retarget_for_local_restore(target)
    return target


def _retarget_for_local_restore(checkpoint: Path) -> None:
    """Patch a downloaded checkpoint copy so ``Algorithm.from_checkpoint``
    restores on the local host without remote runners or a GPU.

    The training config requests ``num_env_runners > 0`` and
    ``num_gpus_per_learner = 1``; on a CPU box restore hangs waiting for those
    resources. Probing needs only the module and the env spec, so the local
    copy is retargeted to a local env runner and an in-process CPU learner.
    The canonical B2 object is untouched.
    """
    import pickle

    args_path = checkpoint / "class_and_ctor_args.pkl"
    recorded = pickle.loads(args_path.read_bytes())
    config_dict = recorded["ctor_args_and_kwargs"][0][0]
    config_dict["num_env_runners"] = 0
    config_dict["num_gpus_per_learner"] = 0
    args_path.write_bytes(pickle.dumps(recorded))


def _run_variant(
    *,
    variant: int,
    seed: int,
    target_steps: int,
    smoke: bool,
    limit: int | None,
    hardware_profile: str,
) -> None:
    run_id = _continued_run_id(variant, seed, target_steps)
    remote, prefix = _remote_prefix(variant, run_id)
    storage = _storage_config(remote)
    client = storage.s3_client()
    experiment = load_experiment(
        f"experiments.{STUDY}.variant_{variant}.experiment"
    )
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=smoke,
        hardware_profile=hardware_profile,
    )
    configure_hardware(context.hardware)
    download_root = context.artifacts_dir / "probe_checkpoints"
    condition = f"variant_{variant}"

    trajectory: list[dict] = []
    last_result = None
    planned = _planned_checkpoints(variant, run_id)
    if limit is not None:
        planned = planned[:limit]
    for entry in planned:
        checkpoint = _download_checkpoint(
            client,
            bucket=storage.bucket,
            prefix=prefix,
            remote_dir=entry["remote_dir"],
            name=entry["name"],
            destination=download_root,
        )
        result, point = _probe_at(
            context,
            checkpoint=checkpoint,
            condition=condition,
            agent_steps=entry["agent_steps"],
        )
        trajectory.append({**point, "checkpoint_name": checkpoint.name})
        last_result = result
        print(
            f"[probe_continued] {condition} steps={entry['agent_steps']} "
            f"mse={point['mse']:.6f} r2={point['probe']['r_squared']:.4f}",
            flush=True,
        )
    if last_result is None:
        raise RuntimeError(f"{condition}: no checkpoints probed")

    results_dir = context.results_dir
    (results_dir / "checkpoint_probe_curve.json").write_text(
        json.dumps(
            {"condition": condition, "checkpoints": trajectory}, indent=2
        )
        + "\n"
    )
    plot_probe(
        last_result,
        title=f"{condition} — continued final",
        path=results_dir / "belief_simplex_final.png",
    )
    summary_path = results_dir / "condition_summary.json"
    if summary_path.is_file():
        summary = _load_json(summary_path)
        summary["checkpoint_probes"] = trajectory
        summary["final_probe"] = last_result.metrics
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", type=int, choices=(1, 2, 3), action="append"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-agent-steps", type=int, default=300_000_000)
    parser.add_argument("--hardware-profile", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for variant in args.variant or (1, 2, 3):
        _run_variant(
            variant=variant,
            seed=args.seed,
            target_steps=args.target_agent_steps,
            smoke=args.smoke,
            limit=args.limit,
            hardware_profile=args.hardware_profile,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
