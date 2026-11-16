"""Continue the manuscript's r7 ``d128_kl0.1`` agents by +300M env steps.

Each of the 8 paper agents (2 r7 runs x seed indices 0-3) resumes from its
final, rollout-bearing update-29 trainer state (30,408,704 steps) with the
r7 PPO config unchanged (constant lr, entropy and clip; the adaptive KL
coefficient, Adam moments and PRNG key continue from the checkpoint). The
budget rounds up to whole 1,048,576-step updates: 287 more updates =
300,941,312 steps, ending at update 316 (331,350,016 lifetime steps).

One process continues one agent. Launch with the source run's results dir
and the seed index::

    ROCKSAMPLE_R7_SEED_INDEX=2 rl-harness \\
      experiments.rocksample_jax_2026_10.ppo_bmax_r7_cont330m.experiment \\
      --seed 42 --resume-from \\
      experiments/rocksample_jax_2026_10/ppo_bmax_r7/results/20261002T182024Z-ad0c3a34

The source ``seed<i>.pkl`` is downloaded from the B2 prefix recorded in the
source ``run_manifest.json`` into ignored ``artifacts/sources/`` (outside this
run's artifacts, so it is not re-uploaded) and verified against the source
durability manifest. Rollout-free checkpoints are saved every 24 updates
(~25M steps) and uploaded to B2 immediately; the final state is resumable.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import jax
from harness.artifacts import RunArtifacts

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import ppo, sweep
from experiments.rocksample_jax_2026_10.baseline import K, N
from experiments.rocksample_jax_2026_10.ppo_bmax_r7 import experiment as r7

SEED_INDEX_ENV = "ROCKSAMPLE_R7_SEED_INDEX"
SOURCE_RUNS = ("20261002T182024Z-ad0c3a34", "20261002T191712Z-7f1dc1ef")
SOURCE_ENV_STEPS = 30_408_704
ADDITIONAL_ENV_STEPS = 300_000_000
CHECKPOINT_EVERY_UPDATES = 24

ARM = r7.ARM
SOURCE_UPDATES = SOURCE_ENV_STEPS // ARM.ppo.batch_size
TOTAL_UPDATES = ARM.num_updates(SOURCE_ENV_STEPS + ADDITIONAL_ENV_STEPS)

SMOKE_ARM = sweep.Arm(
    "smoke_d16", ppo.PPOConfig(num_envs=8, num_steps=128, num_minibatches=2), d_model=16
)


def seed_index() -> int:
    raw = os.environ.get(SEED_INDEX_ENV)
    if raw is None:
        raise ValueError(f"set {SEED_INDEX_ENV} to the r7 seed index (0-3) to continue")
    index = int(raw)
    if not 0 <= index < r7.RECIPE.num_seeds:
        raise ValueError(f"{SEED_INDEX_ENV} must be in [0, {r7.RECIPE.num_seeds}), got {index}")
    return index


def recipe(index: int, *, smoke: bool = False) -> sweep.SweepRecipe:
    arm, start, total = (
        (SMOKE_ARM, 1, 3) if smoke else (ARM, SOURCE_UPDATES, TOTAL_UPDATES)
    )
    return sweep.SweepRecipe(
        arms=(arm,),
        env_steps_per_seed=total * arm.ppo.batch_size,
        num_seeds=1,
        num_keys=r7.RECIPE.num_keys,
        first_seed=index,
        checkpoint_at=tuple(range(start + CHECKPOINT_EVERY_UPDATES, total, CHECKPOINT_EVERY_UPDATES))
        if not smoke
        else (2,),
        upload_checkpoints=True,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_source(source_results: Path, index: int, directory: Path) -> dict:
    """Download and verify ``seed<index>.pkl`` of one r7 run into ``directory``."""
    from harness.storage.b2 import B2StorageConfig

    source_results = Path(source_results)
    if source_results.name not in SOURCE_RUNS:
        raise ValueError(f"--resume-from must be one of the r7 runs {SOURCE_RUNS}")
    remote = json.loads((source_results / "run_manifest.json").read_text())["remote_artifacts"]
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 is not configured; the source checkpoint lives in B2")
    client = config.s3_client()
    manifest = json.loads(
        client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])[
            "Body"
        ].read()
    )
    key = f"{remote['prefix']}/{ARM.name}/seed{index}.pkl"
    record = next(f for f in manifest["files"] if f["key"] == key)
    path = directory / f"seed{index}.pkl"
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != record["size_bytes"]:
        client.download_file(remote["bucket"], key, str(path))
    digest = _sha256(path)
    if digest != record["sha256"] or path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"source checkpoint failed durability verification: {path}")
    with path.open("rb") as stream:
        env_steps = int(pickle.load(stream).env_steps)
    if env_steps != SOURCE_ENV_STEPS:
        raise ValueError(f"source checkpoint has {env_steps} steps, expected {SOURCE_ENV_STEPS}")
    return {
        "run_id": source_results.name,
        "seed_index": index,
        "uri": record["uri"],
        "sha256": digest,
        "size_bytes": record["size_bytes"],
        "env_steps": env_steps,
        "durability_manifest_key": remote["canonical_manifest_key"],
    }


def _smoke_source(context, directory: Path) -> dict:
    """A 1-update smoke trainer state standing in for the r7 checkpoint."""
    env_params = rs.RockSampleParams.from_instance(N, K)
    spec = ppo.make_spec(env_params, d_model=SMOKE_ARM.d_model)
    state = ppo.init(SMOKE_ARM.ppo, env_params, spec, jax.random.key(context.seed))
    state, _ = jax.jit(ppo.make_train_chunk(SMOKE_ARM.ppo, env_params, spec, 1))(state)
    sweep.save_state(directory / "seed0.pkl", state)
    return {"run_id": "smoke", "seed_index": 0, "env_steps": int(state.env_steps)}


def run(context):
    if context.smoke:
        index = 0
        directory = context.artifacts_dir.parent / "source"
        source = _smoke_source(context, directory)
    else:
        if context.resume_from is None:
            raise ValueError("pass --resume-from <ppo_bmax_r7/results/<run-id>>")
        index = seed_index()
        directory = context.experiment_dir / "artifacts" / "sources" / context.resume_from.name
        source = fetch_source(context.resume_from, index, directory)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("continuation_source.json", source)
    base = recipe(index, smoke=context.smoke)
    arm = replace(base.arms[0], resume_from=str(directory))
    return sweep.run(context, replace(base, arms=(arm,)))
