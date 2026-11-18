"""CPU-only five-site extension of the frozen architecture-coarsening analysis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import types

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
ARCH_REV = "30608e20d125402d55e78b6451a2c04a0d36a2ee"
STEER_REV = "45517b13a67d746048a002f9db52ea55a654b219"
ARCH_PATH = str(STUDY.relative_to(ROOT) / "analysis_arch_coarsening/analyze.py")
CORE_PATH = "experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/steering_core.py"
SITES = [f"block_{i}" for i in range(1, 5)] + ["post_final_norm"]
OUTPUT = STUDY / "results/five_site_probes_stage1_seeds42_44"


def git_text(revision, path):
    available = subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                               cwd=ROOT, capture_output=True)
    if available.returncode:
        command = ["git", "fetch", "--no-tags", "--depth=1", "origin", revision]
        try:
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Pinned Git revision {revision} is unavailable locally and could not be "
                f"fetched from origin. Check network access and origin permissions, then "
                f"run `{' '.join(command)}` in {ROOT}."
            ) from exc
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT).decode()


def frozen_module(revision, path):
    source = git_text(revision, path)
    module = types.ModuleType(Path(path).stem)
    module.__file__ = str(ROOT / path)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def replay_sites(spec, params, observations, instrumented_step, empty_maps):
    @jax.jit
    def replay(params, observations):
        def body(carry, inputs):
            t, obs = inputs
            _, carry, sites, _ = instrumented_step(
                spec, params, carry, obs, t, empty_maps(spec.width))
            return carry, sites
        _, sites = jax.lax.scan(
            body, models.init_carry(spec, observations.shape[0]),
            (jnp.arange(observations.shape[1]), jnp.swapaxes(observations, 0, 1)))
        return jnp.swapaxes(sites, 0, 1)
    return replay(params, observations)


def scored(base, features, targets):
    result = base.score(*features, *targets)
    result["r_squared"] = 1 - result["error"]
    return result


def run_one(base, core, variant, seed, index, source_hashes):
    steering_path = str(STUDY.relative_to(ROOT) /
                        f"variant_{variant}_transformer/results/belief_steering_v2_seed{seed}_stable.json")
    steering = json.loads(git_text(STEER_REV, steering_path))
    directory = STUDY / f"variant_{variant}_transformer/results" / steering["checkpoint"]["run_id"]
    recipe = base.read(directory / "resolved_recipe.json")
    manifest = base.read(directory / "run_manifest.json")
    assert recipe["seed"] == seed and recipe["variant"] == variant
    assert manifest["status"] == "completed" and recipe["continued_from"] is None
    checkpoints = base.read(directory / "condition_summary.json")["checkpoint_reports"]
    actual_index = index if index >= 0 else len(checkpoints) + index
    step = checkpoints[actual_index]["checkpoint_step"]
    assert step == (0 if index == 0 else 2012217344)
    output = OUTPUT / f"variant_{variant}_seed{seed}_step{step:010d}.json"
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    path, provenance = base.verified_checkpoint(directory, step)
    if index != 0:
        assert provenance["checkpoint_sha256"] == steering["checkpoint"]["sha256"]
    spec = models.ModelSpec(**recipe["model"])
    params = load_params(path, spec)
    with jax.default_matmul_precision("highest"):
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42), 2))
        raw = [_probe_rollout(spec, make_process(**recipe["process"]), params, key,
                              -(-n // 960), 1024) for key, n in zip(keys, (60000, 80000))]
        features = []
        parity = []
        for rows, n in zip(raw, (60000, 80000)):
            sites = np.asarray(replay_sites(spec, params, rows["obs"],
                                            core.instrumented_step, core.empty_maps))
            difference = float(np.max(np.abs(sites[:, :, -1] - np.asarray(rows["embedding"])) ))
            np.testing.assert_allclose(sites[:, :, -1], rows["embedding"], atol=1e-6, rtol=1e-6)
            parity.append(difference)
            features.append(base.flatten(sites, n))
        beliefs = [base.flatten(r["belief"], n) for r, n in zip(raw, (60000, 80000))]
        coarse = [base.flatten(base.merged_filter(r["obs"], recipe["process"]), n)
                  for r, n in zip(raw, (60000, 80000))]
        targets = {"b": beliefs, "c": coarse,
                   "rho": [b[:, 2:3] for b in beliefs],
                   "delta": [(b[:, 0:1] - b[:, 1:2]) for b in beliefs]}
        scores = {site: {name: scored(base, [x[:, i] for x in features], target)
                         for name, target in targets.items()}
                  for i, site in enumerate(SITES)}
        original_features = [base.flatten(r["embedding"], n) for r, n in zip(raw, (60000, 80000))]
        reproduction = {name: base.score(*original_features, *targets[name]) for name in ("b", "c")}
    comparison = {}
    for name in ("b", "c"):
        new = scores["post_final_norm"][name]["error"]
        reference = reproduction[name]["error"]
        assert abs(new - reference) < 1e-7, (name, new, reference)
        comparison[name] = {"five_site_error": new, "new_final_norm_only_error": reference,
                            "absolute_difference": abs(new - reference)}
    if seed == 42:
        archived_path = str(STUDY.relative_to(ROOT) /
                            f"analysis_arch_coarsening/results/variant_{variant}_transformer_{step:010d}.json")
        archived = json.loads(git_text(ARCH_REV, archived_path))
        for name in ("b", "c"):
            reference = archived["scores"][name]["error"]
            difference = abs(scores["post_final_norm"][name]["error"] - reference)
            comparison[name].update(archived_error=reference, archived_absolute_difference=difference)
            assert difference < 1e-5, (variant, step, name, difference)
    result = {"variant": variant, "seed": seed, "checkpoint_step": step,
              "checkpoint_index": actual_index, "run_id": directory.name,
              "results_path": str(directory.relative_to(ROOT)), "training_git": manifest["git"],
              **provenance, "c_is_exact_quotient": variant in (1, 2),
              "c_definition": "merged-token reward-state posterior; Full-State is a control, not an exact quotient",
              "protocol": {"fit_rows": 60000, "test_rows": 80000, "episode_length": 1024,
                           "warmup": 64, "key": "split(fold_in(PRNGKey(42), 2))",
                           "policy": "own-checkpoint greedy", "ridge": 1e-6,
                           "matmul_precision": "highest", "sites": SITES},
              "scores": scores, "final_norm_comparison": comparison,
              "replay_final_norm_max_absolute_difference": parity,
              "steering_coordinate_r2_reference": steering["probes"]["exclusion_8"] if index != 0 else None,
              "source": {"architecture_revision": ARCH_REV, "steering_revision": STEER_REV,
                         "source_sha256": source_hashes,
                         "experiment_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                         "harness_revision": subprocess.check_output(["git", "-C", str(ROOT.parent / "rl-harness"), "rev-parse", "HEAD"]).decode().strip(),
                         "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "runtime": {"jax": jax.__version__, "numpy": np.__version__, "devices": str(jax.devices())},
              "elapsed_seconds": time.monotonic() - started}
    base.write(output, result)
    print(json.dumps({"file": str(output.relative_to(ROOT)), "final_norm": comparison,
                      "seconds": result["elapsed_seconds"]}), flush=True)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base, arch_hash = frozen_module(ARCH_REV, ARCH_PATH)
    core, core_hash = frozen_module(STEER_REV, CORE_PATH)
    for variant in (1, 2, 3):
        for seed in (42, 43, 44):
            for index in (0, -1):
                run_one(base, core, variant, seed, index,
                        {ARCH_PATH: arch_hash, CORE_PATH: core_hash})


if __name__ == "__main__":
    main()
