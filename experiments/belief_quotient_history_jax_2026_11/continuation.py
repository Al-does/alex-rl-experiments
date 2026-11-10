"""Continue a finished JAX history run from its final parameter checkpoint.

The parent run is located through its checked-in compact results for the
requested seed (``resolved_recipe.json`` + ``condition_summary.json`` +
``run_manifest.json``); its final ``step_<N>.npz`` is read from the local
``artifacts/`` tree when present, otherwise fetched from the B2 prefix in the
manifest. Checkpoints hold parameters only, so the continuation restarts Adam
moments (the recipe's constant learning rate and entropy are unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models


def locate_parent_run(parent_leaf_dir: Path, *, study: str, seed: int) -> dict:
    matches = []
    for manifest_path in sorted(Path(parent_leaf_dir).glob("results/*/run_manifest.json")):
        recipe_path = manifest_path.with_name("resolved_recipe.json")
        summary_path = manifest_path.with_name("condition_summary.json")
        if not recipe_path.is_file() or not summary_path.is_file():
            continue
        recipe = json.loads(recipe_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if recipe.get("smoke") or recipe.get("study") != study or int(recipe["seed"]) != int(seed):
            continue
        if manifest.get("status") != "completed":
            continue
        matches.append((manifest, recipe, json.loads(summary_path.read_text())))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one completed seed-{seed} {study} run under "
                         f"{parent_leaf_dir}/results, found {len(matches)}")
    manifest, recipe, summary = matches[0]
    final_step = int(summary["training"]["env_steps"])
    remote = manifest.get("remote_artifacts") or {}
    return {
        "run_id": manifest["run_id"],
        "study": recipe["study"],
        "condition": recipe["condition"],
        "architecture": recipe["architecture"],
        "seed": int(recipe["seed"]),
        "final_step": final_step,
        "checkpoint_file": f"checkpoints/step_{final_step:010d}.npz",
        "final_reward_per_step": summary["training"].get("final_reward_per_step"),
        "experiment_commit": (manifest.get("git") or {}).get("commit"),
        "b2_bucket": remote.get("bucket"),
        "b2_prefix": remote.get("prefix"),
    }


def fetch_parent_checkpoint(parent: dict, parent_leaf_dir: Path, destination: Path) -> Path:
    local = Path(parent_leaf_dir) / "artifacts" / parent["run_id"] / parent["checkpoint_file"]
    if local.is_file():
        return local
    destination = Path(destination)
    if destination.is_file():
        return destination
    if not parent.get("b2_bucket") or not parent.get("b2_prefix"):
        raise ValueError(f"parent run {parent['run_id']} has no local checkpoint or B2 prefix")
    from harness.storage.b2 import B2StorageConfig

    storage = B2StorageConfig.from_env()
    if storage is None:
        raise RuntimeError("B2 credentials are required to fetch the parent checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage.s3_client().download_file(
        parent["b2_bucket"], f"{parent['b2_prefix']}/{parent['checkpoint_file']}", str(destination))
    return destination


def load_params(path: Path, spec) -> dict:
    """Rebuild the parameter pytree saved by ``training._save`` (keystr -> array)."""

    template = models.init_params(spec, jax.random.PRNGKey(0))
    with np.load(path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    missing = [jax.tree_util.keystr(k) for k, _ in leaves if jax.tree_util.keystr(k) not in arrays]
    if missing or len(arrays) != len(leaves):
        raise ValueError(f"checkpoint {path} does not match the model: missing {missing[:3]}, "
                         f"{len(arrays)} saved vs {len(leaves)} expected leaves")
    values = []
    for key, leaf in leaves:
        value = arrays[jax.tree_util.keystr(key)]
        if value.shape != leaf.shape:
            raise ValueError(f"shape mismatch at {jax.tree_util.keystr(key)}: {value.shape} vs {leaf.shape}")
        values.append(jax.numpy.asarray(value, dtype=leaf.dtype))
    return jax.tree_util.tree_unflatten(treedef, values)


def load_parent_checkpoint(context, *, parent_leaf_dir: Path, study: str, spec) -> dict:
    parent = locate_parent_run(parent_leaf_dir, study=study, seed=context.seed)
    path = fetch_parent_checkpoint(
        parent, parent_leaf_dir, context.artifacts_dir / "parent" / Path(parent["checkpoint_file"]).name)
    return {"params": load_params(path, spec), "step_offset": parent["final_step"],
            "parent": {**parent, "checkpoint_path": str(path)}}
