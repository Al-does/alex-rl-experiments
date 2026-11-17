"""CPU-only seed-42 b versus merged-token c probes, following paper PR #23."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def merged_filter(observations, process):
    """Filter complete prefixes with A/B merged, retaining three latent states."""
    obs = np.asarray(observations)
    if obs.ndim != 3 or not obs.shape[0] or not obs.shape[1] or obs.shape[2] != 6:
        raise ValueError("expected nonempty episode histories with six observation features")
    emission = np.asarray(process["emission"], dtype=np.float64)
    emission = np.stack((emission[:, :2].sum(1), emission[:, 2]), axis=1)
    transitions = np.asarray(process["transitions"], dtype=np.float64)
    tokens = (obs[..., :3].argmax(-1) == 2).astype(int)
    actions = obs[..., 3:].argmax(-1)
    belief = np.broadcast_to(np.asarray(process["initial"], dtype=np.float64), (len(obs), 3)).copy()
    result = np.empty(obs.shape[:2] + (1,), dtype=np.float64)
    for t in range(obs.shape[1]):
        if t:
            belief = np.einsum("bi,bij->bj", belief, transitions[actions[:, t]])
        belief *= emission[:, tokens[:, t]].T
        normalizer = belief.sum(1, keepdims=True)
        if np.any(normalizer <= 0):
            raise ValueError("impossible merged observation history")
        belief /= normalizer
        result[:, t, 0] = belief[:, 2]
    return result


def flatten(value, rows):
    value = np.asarray(value)[:, 64:]
    return value.reshape(-1, *value.shape[2:])[:rows]


def score(x, z, y, target):
    from analysis.probes.linear import fit_affine_probe, global_mse_metrics, r2_score
    weight, bias = fit_affine_probe(x, y, ridge=1e-6)
    prediction = np.asarray(z, dtype=np.float64) @ weight + bias
    return {**global_mse_metrics(prediction, target), "error": 1 - r2_score(prediction, target)}


def verified_checkpoint(directory, step):
    from harness.storage.b2 import B2StorageConfig
    remote = read(directory / "run_manifest.json")["remote_artifacts"]
    client = B2StorageConfig.from_env().s3_client()
    raw = client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read()
    relative = f"checkpoints/step_{step:010d}.npz"
    row, = [r for r in json.loads(raw)["files"] if r["relative_path"] == relative]
    path = STUDY / "artifacts" / "arch_coarsening" / directory.name / Path(relative).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        client.download_file(remote["bucket"], row["key"], str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    expected_size = row.get("size_bytes", row.get("size"))
    if expected_size is None or size != expected_size or digest != row["sha256"]:
        raise ValueError(f"durability verification failed: {path}; manifest entry {row}")
    return path, {"checkpoint_uri": f"s3://{remote['bucket']}/{row['key']}",
                  "checkpoint_sha256": digest, "checkpoint_bytes": size,
                  "manifest_key": remote["canonical_manifest_key"],
                  "manifest_sha256": hashlib.sha256(raw).hexdigest()}


def run(directory, index):
    import jax
    from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
    from experiments.belief_quotient_history_jax_2026_11.env import make_process
    from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
    from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout
    recipe = read(directory / "resolved_recipe.json")
    manifest = read(directory / "run_manifest.json")
    report = read(directory / "condition_summary.json")["checkpoint_reports"][index]
    step = report["checkpoint_step"]
    output = HERE / "results" / f"{directory.parent.parent.name}_{step:010d}.json"
    if output.exists():
        return read(output)
    start = time.monotonic()
    path, provenance = verified_checkpoint(directory, step)
    spec = ModelSpec(**recipe["model"])
    params = load_params(path, spec)
    site = "residual" if spec.kind == "gru" else "embedding"
    with jax.default_matmul_precision("highest"):
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42), 2))
        raw = [_probe_rollout(spec, make_process(**recipe["process"]), params, key,
                             -(-n // 960), 1024) for key, n in zip(keys, (60000, 80000))]
        features = [flatten(r[site], n) for r, n in zip(raw, (60000, 80000))]
        beliefs = [flatten(r["belief"], n) for r, n in zip(raw, (60000, 80000))]
        coarse = [flatten(merged_filter(r["obs"], recipe["process"]), n)
                  for r, n in zip(raw, (60000, 80000))]
        scores = {"b": score(*features, *beliefs), "c": score(*features, *coarse)}
    result = {"architecture": recipe["architecture"], "condition": recipe["condition"],
              "seed": 42, "run_id": directory.name, "results_path": str(directory.relative_to(ROOT)),
              "training_git": manifest["git"], "checkpoint_index": index, "checkpoint_step": step,
              **provenance, "representation": site, "model": recipe["model"],
              "c_is_exact_quotient": recipe["variant"] in (1, 2), "scores": scores,
              "greedy_reward_per_step": float(flatten(raw[1]["reward"], 80000).mean()),
              "saved_greedy_reward_per_step": report["policy"]["reward_per_step"],
              "elapsed_seconds": time.monotonic() - start,
              "runtime": {"jax": jax.__version__, "numpy": np.__version__}}
    write(output, result)
    print(recipe["architecture"], recipe["condition"], index, scores,
          result["greedy_reward_per_step"], result["elapsed_seconds"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("transformer", "gru128", "gru92", "memoryless"), required=True)
    parser.add_argument("--curves", action="store_true")
    args = parser.parse_args()
    for variant in (1, 2, 3):
        directories = [p.parent for p in (STUDY / f"variant_{variant}_{args.architecture}" / "results").glob("*/resolved_recipe.json")
                       if read(p)["seed"] == 42 and not read(p)["smoke"]]
        directory, = directories
        assert read(directory / "run_manifest.json")["status"] == "completed"
        assert read(directory / "resolved_recipe.json")["continued_from"] is None
        count = len(read(directory / "condition_summary.json")["checkpoint_reports"])
        indices = sorted(set(range(0, count, 5)) | {1, 2, 3, 4, count - 1}) if args.curves else [count - 1]
        for index in indices:
            run(directory, index)


if __name__ == "__main__":
    main()
