"""Continue a finished history-dependent run from its final checkpoint.

The parent run is located through its checked-in compact results (the
``run_manifest.json``/``resolved_recipe.json`` pair for the requested seed),
and its final Tune checkpoint is fetched from the B2 prefix recorded there
(or reused from a local ``artifacts/`` tree when present). The returned
source is restored into the fresh Algorithm by ``training.run_study`` via
``Algorithm.restore_from_path`` so weights, optimizer state, lifetime step
counters, and the training iteration all continue from the parent.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

from harness.storage.b2 import B2StorageConfig


@dataclass(frozen=True)
class ContinuationSource:
    checkpoint_path: Path
    agent_steps: int
    training_iteration: int
    parent: dict = field(default_factory=dict)


def locate_parent_run(parent_leaf_dir: Path, *, seed: int) -> dict:
    matches = []
    for manifest_path in sorted(parent_leaf_dir.glob("results/*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        recipe_path = manifest_path.with_name("resolved_recipe.json")
        summary_path = manifest_path.with_name("tune_summary.json")
        if not recipe_path.is_file() or not summary_path.is_file():
            continue
        recipe = json.loads(recipe_path.read_text())
        if recipe.get("smoke") or int(recipe["seed"]) != int(seed):
            continue
        if manifest.get("status") != "completed":
            continue
        matches.append((manifest_path, manifest, recipe, json.loads(summary_path.read_text())))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one completed seed-{seed} parent run under "
            f"{parent_leaf_dir}/results, found {len(matches)}"
        )
    manifest_path, manifest, recipe, summary = matches[0]
    trials = summary.get("trials", [])
    if len(trials) != 1 or trials[0].get("error") or not trials[0].get("checkpoint"):
        raise ValueError(f"parent run {manifest['run_id']} has no clean final checkpoint")
    trial = trials[0]
    metrics = trial["metrics"]
    run_id = manifest["run_id"]
    checkpoint = Path(trial["checkpoint"])
    parts = checkpoint.parts
    anchor = parts.index(run_id) if run_id in parts else -1
    if anchor < 0 or parts[anchor - 1] != "artifacts":
        raise ValueError(f"cannot relate {checkpoint} to artifacts/{run_id}")
    relative = Path(*parts[anchor + 1 :])
    remote = manifest.get("remote_artifacts") or {}
    return {
        "run_id": run_id,
        "study": recipe["study"],
        "condition": recipe["condition"],
        "architecture": recipe["architecture"],
        "seed": int(recipe["seed"]),
        "total_env_steps": int(recipe["total_env_steps"]),
        "results_dir": str(manifest_path.parent),
        "experiment_commit": (manifest.get("git") or {}).get("commit"),
        "library_commit": ((manifest.get("git") or {}).get("library") or {}).get("commit"),
        "checkpoint_relative_path": str(relative),
        "local_checkpoint_path": str(checkpoint),
        "agent_steps": int(metrics["env_runners/num_env_steps_sampled_lifetime"]),
        "training_iteration": int(metrics["training_iteration"]),
        "b2_bucket": remote.get("bucket"),
        "b2_prefix": remote.get("prefix"),
        "b2_endpoint": remote.get("endpoint"),
    }


def fetch_parent_checkpoint(parent: dict, destination: Path) -> Path:
    marker = destination / "rllib_checkpoint.json"
    if marker.is_file():
        return destination
    local = Path(parent["local_checkpoint_path"])
    try:
        if (local / "rllib_checkpoint.json").is_file():
            return local
    except OSError:
        pass
    if not parent.get("b2_prefix") or not parent.get("b2_bucket"):
        raise ValueError(f"parent run {parent['run_id']} has no B2 prefix to fetch from")
    storage = B2StorageConfig.from_env()
    if storage is None:
        raise RuntimeError("B2 credentials are required to fetch the parent checkpoint")
    client = storage.s3_client()
    key_prefix = f"{parent['b2_prefix']}/{parent['checkpoint_relative_path']}/"
    paginator = client.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=parent["b2_bucket"], Prefix=key_prefix)
        for obj in page.get("Contents", [])
    ]
    if not keys:
        raise RuntimeError(f"no objects under s3://{parent['b2_bucket']}/{key_prefix}")
    destination.mkdir(parents=True, exist_ok=True)
    for key in keys:
        target = destination / key[len(key_prefix) :]
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(parent["b2_bucket"], key, str(target))
    if not marker.is_file():
        raise RuntimeError(f"downloaded checkpoint lacks rllib_checkpoint.json: {destination}")
    return destination


def prepare_continuation(context, *, parent_leaf_dir: Path) -> ContinuationSource:
    if context.seed is None:
        raise ValueError("continuation requires a resolved seed")
    parent = locate_parent_run(Path(parent_leaf_dir), seed=context.seed)
    destination = context.artifacts_dir / "parent_checkpoint" / Path(parent["checkpoint_relative_path"]).name
    checkpoint = fetch_parent_checkpoint(parent, destination)
    return ContinuationSource(
        checkpoint_path=checkpoint,
        agent_steps=parent["agent_steps"],
        training_iteration=parent["training_iteration"],
        parent=parent,
    )
