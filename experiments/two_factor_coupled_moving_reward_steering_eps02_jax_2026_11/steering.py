"""XOR and moving-reward belief steering on the PR #234 Joint Reward agents.

The protocol, rewards, action maps and update rule are those of
``two_factor_coupled_moving_reward_steering_v2_jax_2026_11``. Only the process
changes: the agents were trained with the emission-epsilon 0.02 matrix
(``belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11``), so rollouts,
exact beliefs and action maps are built from that process. The transitions are
those of v2, so the known-state action values are unchanged; the exact belief
is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import harness
from harness.storage.b2 import B2StorageConfig

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process
from experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11 import steering as coupled
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11 import steering as base

STUDY_DIR = Path(__file__).resolve().parent
SOURCE_STUDY = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11"
SOURCE_LEAF = "reward_both_transformer"
CONDITION = "reward_both"
FINAL_STEP = 2_012_217_344


def source_process():
    return factor_process(CONDITION, environment)


def source_runs() -> list[dict]:
    runs = []
    leaf = base.REPO / "experiments" / SOURCE_STUDY / SOURCE_LEAF
    for manifest in sorted(leaf.glob("results/*/run_manifest.json")):
        data = json.loads(manifest.read_text())
        runs.append({"study": SOURCE_STUDY, "leaf": SOURCE_LEAF, "condition": CONDITION,
                     "seed": data["runtime"]["seed"], "run_id": data["run_id"],
                     "results": manifest.parent.relative_to(base.REPO),
                     "remote_prefix": data["remote_artifacts"]["prefix"]})
    return sorted(runs, key=lambda run: run["seed"])


def fetch_checkpoint(run: dict, checkpoints: Path) -> Path:
    """Final checkpoint, downloaded from B2 and matched to the durability manifest."""
    relative = f"checkpoints/step_{FINAL_STEP:010d}.npz"
    path = checkpoints / f"seed{run['seed']}__{run['run_id']}__step_{FINAL_STEP:010d}.npz"
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2_* settings are required to verify checkpoints")
    client = config.s3_client()
    manifest = json.loads(client.get_object(
        Bucket=config.bucket, Key=f"{run['remote_prefix']}/metadata/durability_manifest.json",
    )["Body"].read())
    (expected,) = (f["sha256"] for f in manifest["files"] if f["relative_path"] == relative)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(config.bucket, f"{run['remote_prefix']}/{relative}", str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{path}: SHA-256 does not match the durability manifest")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=STUDY_DIR / "artifacts" / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=STUDY_DIR / "results")
    parser.add_argument("--tune-episodes", type=int, default=64)
    parser.add_argument("--confirm-episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-seeds", type=int, nargs="+", default=None,
                        help="training seeds to run (default: all)")
    args = parser.parse_args()
    revisions = {"experiment": base._git(STUDY_DIR),
                 "harness": base._git(Path(harness.__file__).resolve().parent)}
    process = source_process()
    for run in source_runs():
        if args.agent_seeds and run["seed"] not in args.agent_seeds:
            continue
        output = args.output_dir / f"eps02_ctx32__{CONDITION}__seed{run['seed']}.json"
        if output.exists():
            raise FileExistsError(output)
        report = coupled.analyze(
            {**run, "results": base.REPO / run["results"]}, fetch_checkpoint(run, args.checkpoints),
            tune_episodes=args.tune_episodes, confirm_episodes=args.confirm_episodes,
            seed=args.seed, process=process)
        report["source"]["results"] = str(run["results"])
        report["environment"] = environment.environment_metadata()
        report["revisions"] = revisions
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
