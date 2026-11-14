"""Post-hoc manuscript probe battery for the 24 agents in PR #219.

No training or historical results are modified. Checkpoints are verified against
their canonical B2 manifests; raw geometry and restart caches stay in artifacts/.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from analysis.probes.linear import fit_affine_probe, global_mse_metrics
from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_history_jax_2026_11.reinforce import behaviour_log_probs, rollout
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment
from harness.storage.b2 import B2StorageConfig

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDIES = {
    "no_eps": "belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11",
    "eps05": "belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11",
}
ARCHITECTURES = ("transformer", "memoryless", "gru128", "gru92")
PROTOCOL = {
    "seed": 42, "fit_rows": 60000, "test_rows": 80000, "warmup": 64,
    "context_robustness_warmup": 128, "episode_length": 1024, "ridge": 1e-6,
    "control_rows": 20000, "control_warmup": 128, "ab_episodes": 64,
    "ab_warmup": 128, "bootstrap_resamples": 2000,
    "representation": "post_final_layer_norm_embedding",
    "matmul_precision": "highest", "probe_solver": "numpy_float64_svd_ridge",
    "fit_test_streams": [510, 511], "control_streams": [1100, 1101],
    "ab_stream": 1200, "primary_policy": "greedy",
    "ab_policy": "sampled_training_behaviour_T1.5_with_arm_epsilon",
    "uncertainty": "episode bootstrap for A/B TV only; one training seed",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def discover(exploration, architecture, variant):
    leaf = ROOT / "experiments" / STUDIES[exploration] / f"variant_{variant}_{architecture}"
    runs = []
    for path in sorted(leaf.glob("results/*/run_manifest.json")):
        manifest = json.loads(path.read_text())
        recipe = json.loads(path.with_name("resolved_recipe.json").read_text())
        if manifest["status"] == "completed" and recipe["seed"] == 42 and not recipe["smoke"]:
            runs.append((path, manifest, recipe))
    if len(runs) != 1:
        raise ValueError(f"expected one seed-42 run: {leaf}; found {len(runs)}")
    path, manifest, recipe = runs[0]
    summary_path = path.with_name("condition_summary.json")
    summary = json.loads(summary_path.read_text())
    process = environment.ladder_process(variant)
    for name, actual in process._asdict().items():
        np.testing.assert_allclose(actual, recipe["process"][name], atol=1e-7, rtol=1e-7)
    return leaf, manifest, recipe, summary, sha256(path), sha256(summary_path)


class Checkpoints:
    def __init__(self, manifest, cache):
        storage = B2StorageConfig.from_env()
        if storage is None:
            raise RuntimeError("B2 credentials are required")
        self.client = storage.s3_client()
        self.remote = manifest["remote_artifacts"]
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        response = self.client.get_object(Bucket=self.remote["bucket"],
                                          Key=self.remote["canonical_manifest_key"])
        raw = response["Body"].read()
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        self.entries = {row["relative_path"]: row for row in json.loads(raw)["files"]}

    def get(self, step):
        name = f"checkpoints/step_{step:010d}.npz"
        entry = self.entries[name]
        path = self.cache / Path(name).name
        if not path.exists() or sha256(path) != entry["sha256"]:
            self.client.download_file(self.remote["bucket"], entry["key"], str(path))
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"checkpoint integrity failed: {path}")
        return path, entry["sha256"]


@partial(jax.jit, static_argnums=(0, 4, 5, 6, 7))
def collect(spec, process, params, key, episodes, length, greedy=True, epsilon=0.0):
    return rollout(spec, process, params, key, episodes, length, greedy=greedy,
                   track_belief=True, temperature=1.5, epsilon=epsilon)


@partial(jax.jit, static_argnums=(0,))
def replay(spec, params, observations):
    carry = models.init_carry(spec, observations.shape[0])

    def body(carry, inputs):
        t, obs = inputs
        logits, _, embedding, carry = models.step(spec, params, carry, obs, t)
        return carry, (logits, embedding)

    _, (logits, embedding) = jax.lax.scan(
        body, carry, (jnp.arange(observations.shape[1]), jnp.swapaxes(observations, 0, 1)))
    return jnp.swapaxes(logits, 0, 1), jnp.swapaxes(embedding, 0, 1)


def replay_filter(observations, process, coarse=False):
    obs = np.asarray(observations)
    tokens = obs[..., :3].argmax(-1)
    actions = obs[..., 3:].argmax(-1)
    initial, transitions, emission = [np.asarray(x, dtype=np.float64) for x in
                                     (process.initial, process.transitions, process.emission)]
    transitions = transitions / transitions.sum(-1, keepdims=True)
    emission = emission / emission.sum(-1, keepdims=True)
    if coarse:
        if not np.allclose(transitions[:, 0, 2], transitions[:, 1, 2], atol=1e-8):
            raise ValueError("this condition has no exact Reward-State quotient")
        transitions = np.stack([np.stack([1 - transitions[:, 0, 2], transitions[:, 0, 2]], -1),
                                np.stack([1 - transitions[:, 2, 2], transitions[:, 2, 2]], -1)], 1)
        emission = np.array([[emission[0, :2].sum(), emission[0, 2]],
                             [emission[2, :2].sum(), emission[2, 2]]])
        initial = np.array([initial[:2].sum(), initial[2]])
        tokens = (tokens == 2).astype(int)
    belief = np.broadcast_to(initial / initial.sum(), (len(obs), len(initial))).copy()
    result = np.empty((*tokens.shape, len(initial)))
    for t in range(obs.shape[1]):
        if t:
            belief = np.einsum("bi,bij->bj", belief, transitions[actions[:, t]])
        belief *= emission[:, tokens[:, t]].T
        belief /= belief.sum(-1, keepdims=True)
        result[:, t] = belief
    return result


def rows(value, warmup, count):
    value = np.asarray(value)[:, warmup:]
    return value.reshape(-1, *value.shape[2:])[:count]


def targets(belief, coarse=None):
    out = {"belief": belief, "rho": belief[:, 2:3],
           "delta": belief[:, :1] - belief[:, 1:2]}
    if coarse is not None:
        out["coarse"] = coarse[:, 1:2]
    return out


def score(features_fit, features_test, target_fit, target_test):
    weight, bias = fit_affine_probe(features_fit, target_fit, ridge=1e-6)
    predicted = features_test @ weight + bias
    return global_mse_metrics(predicted, target_test), predicted


def control_inputs(observations, kind, rng):
    obs = np.asarray(observations)
    out = obs.copy()
    if kind == "shuffle":
        for i in range(len(obs)):
            out[i, 1:] = obs[i, rng.permutation(np.arange(1, obs.shape[1]))]
    elif kind == "random_input":
        reset, other = obs[:, 0], obs[:, 1:].reshape(-1, obs.shape[-1])
        out[:, 0] = reset[rng.integers(len(reset), size=len(obs))]
        out[:, 1:] = other[rng.integers(len(other), size=out[:, 1:].shape[:2])]
    else:
        raise ValueError(kind)
    return out


def ab_inputs(observations, rng):
    out = np.asarray(observations).copy()
    original = out[..., :3].argmax(-1)
    token = np.where(original == 2, 2, rng.integers(2, size=original.shape))
    out[..., :3] = np.eye(3, dtype=out.dtype)[token]
    return out, (token != original), original != 2


def episode_interval(values, seed=1400):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(-1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "episodes": len(values)}


@partial(jax.jit, static_argnums=(0, 4, 5))
def ab_closed_loop(spec, process, params, key, episodes, length):
    reset_key, key = jax.random.split(key)
    state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    carry = models.init_carry(spec, episodes)

    def body(value, t):
        state, carry, key = value
        key, _, step_key = jax.random.split(key, 3)
        obs = jax.vmap(env.observe, (None, 0))(process, state)
        altered = jax.random.randint(jax.random.fold_in(key, 99), (episodes,), 0, 2)
        altered = jnp.where(state.token == 2, 2, altered)
        obs = obs.at[:, :3].set(jax.nn.one_hot(altered, 3))
        logits, _, _, carry = models.step(spec, params, carry, obs, t)
        action = logits.argmax(-1)
        state, reward, _ = jax.vmap(env.step, (None, 0, 0, 0))(
            process, jax.random.split(step_key, episodes), state, action)
        return (state, carry, key), (action, reward)

    _, (actions, rewards) = jax.lax.scan(body, (state, carry, key), jnp.arange(length))
    return actions.T, rewards.T


def analyze_checkpoint(spec, process, params, *, variant, epsilon, controls, final, smoke):
    length = 1024
    fit_count, test_count = (1024, 1024) if smoke else (60000, 80000)
    # Enough complete prefixes for both historical 64- and full-context 128-step masks.
    fit_n, test_n = [-(-n // (length - 128)) for n in (fit_count, test_count)]
    streams = []
    for stream, count in ((510, fit_n), (511, test_n)):
        data = collect(spec, process, params, jax.random.PRNGKey(stream), count, length)
        data = jax.tree_util.tree_map(np.asarray, data)
        exact = replay_filter(data["obs"], process)
        error = float(np.max(np.abs(exact - data["belief"])))
        if error > 2e-5:
            raise ValueError(f"decision-time filter check failed: {error}")
        coarse = replay_filter(data["obs"], process, coarse=True) if variant == 2 else None
        streams.append((data, exact, coarse, error))
    report = {"filter_max_abs_error": max(s[3] for s in streams), "targets": {}}
    predictions = None
    for warmup in (64, 128):
        fit, test = [targets(rows(s[1], warmup, n),
                            rows(s[2], warmup, n) if s[2] is not None else None)
                     for s, n in zip(streams, (fit_count, test_count))]
        features_fit, features_test = [rows(s[0]["embedding"], warmup, n)
                                      for s, n in zip(streams, (fit_count, test_count))]
        scores = {}
        for name in fit:
            scores[name], predicted = score(features_fit, features_test, fit[name], test[name])
            if warmup == 64 and name == "belief":
                predictions = predicted
        report["targets" if warmup == 64 else "full_context128_targets"] = scores
        if variant == 2:
            report["coarse_rho_max_abs_difference"] = max(
                float(np.max(np.abs(fit["coarse"] - fit["rho"]))),
                float(np.max(np.abs(test["coarse"] - test["rho"]))))
    test_data = streams[1][0]
    report["policy"] = {
        "reward_per_step": float(rows(test_data["reward"], 64, test_count).mean()),
        "action_fractions": (np.bincount(rows(test_data["action"], 64, test_count), minlength=3)
                             / test_count).tolist(),
    }
    if final:
        report["simplex_projection_negative_fraction"] = float(np.any(
            predictions + (1 - predictions.sum(-1, keepdims=True)) / 3 < 0, axis=-1).mean())
        geometry = {"target": rows(streams[1][1], 64, test_count), "decoded": predictions}
    else:
        geometry = None
    del streams
    if controls:
        budget = 1024 if smoke else 20000
        episodes = -(-budget // (length - 128))
        control_streams = []
        for seed in (1100, 1101):
            data = collect(spec, process, params, jax.random.PRNGKey(seed), episodes, length)
            control_streams.append({k: np.asarray(data[k]) for k in ("obs", "belief")})
        report["controls"] = {}
        for kind in ("shuffle", "random_input"):
            features = []
            for i, data in enumerate(control_streams):
                obs = control_inputs(data["obs"], kind, np.random.default_rng(1110 + i))
                _, embedding = replay(spec, params, jnp.asarray(obs))
                features.append(rows(embedding, 128, budget))
            yy = [rows(data["belief"], 128, budget) for data in control_streams]
            report["controls"][kind], _ = score(*features, *yy)
    if variant == 2:
        episodes = 2 if smoke else 64
        key = jax.random.PRNGKey(1200)
        sampled = collect(spec, process, params, key, episodes, length, False, epsilon)
        obs = np.asarray(sampled["obs"])
        altered, changed, eligible = ab_inputs(obs, np.random.default_rng(1201))
        logits, _ = replay(spec, params, jnp.asarray(obs))
        alt_logits, _ = replay(spec, params, jnp.asarray(altered))
        probabilities = [np.asarray(jnp.exp(behaviour_log_probs(x, 1.5, epsilon)))
                         for x in (logits, alt_logits)]
        tv = np.abs(probabilities[0] - probabilities[1]).sum(-1) / 2
        report["ab_sensitivity"] = {
            "tv": episode_interval(tv[:, 128:].mean(-1)),
            "greedy_disagreement": episode_interval(
                (np.asarray(logits).argmax(-1) != np.asarray(alt_logits).argmax(-1))[:, 128:].mean(-1)),
            "ab_replacement_fraction": float(changed.sum() / eligible.sum()),
            "factual_actions_preserved": True,
        }
        if final:
            factual = collect(spec, process, params, key, episodes, length)
            alt_actions, alt_reward = ab_closed_loop(spec, process, params, key, episodes, length)
            factual_reward = np.asarray(factual["reward"])
            report["ab_closed_loop"] = {
                "greedy_action_disagreement": float((np.asarray(factual["action"]) != alt_actions).mean()),
                "factual_reward": episode_interval(factual_reward.mean(-1)),
                "randomized_reward": episode_interval(np.asarray(alt_reward).mean(-1)),
                "paired_reward_difference": episode_interval((np.asarray(alt_reward) - factual_reward).mean(-1)),
                "coupling": "same reset and transition/emission PRNG innovations; separate policy actions",
                "warmup": 0,
            }
    return report, geometry


def run_arm(exploration, architecture, variant, smoke=False):
    start = time.monotonic()
    arm = f"{exploration}_variant_{variant}_{architecture}"
    leaf, manifest, recipe, summary, manifest_hash, summary_hash = discover(exploration, architecture, variant)
    cache = HERE / (".smoke" if smoke else "artifacts") / "paper_probes" / arm
    result = HERE / (".smoke" if smoke else "results") / "paper_probes" / f"{arm}.json"
    process = environment.ladder_process(variant)
    spec = models.ModelSpec(**recipe["model"])
    checkpoints = Checkpoints(manifest, HERE / "artifacts" / "paper_probe_checkpoints" / arm)
    archived = summary["checkpoint_reports"]
    chosen = [archived[0], archived[-1]] if smoke else archived
    steps = [int(p["checkpoint_step"]) for p in chosen]
    batch = int(recipe["training"]["num_envs"]) * int(recipe["training"]["episode_length"])
    control_steps = {0, steps[-1]} | {batch * 2**k for k in range(7)}
    epsilon = float(recipe["training"].get("exploration_epsilon", 0))
    protocol = {**PROTOCOL, "smoke": smoke, "exploration_epsilon": epsilon}
    fingerprint = hashlib.sha256(json.dumps({"protocol": protocol, "source": sha256(__file__),
                                            "manifest": manifest_hash}, sort_keys=True).encode()).hexdigest()
    reports = []
    for archived_report in chosen:
        step = int(archived_report["checkpoint_step"])
        path, weight_hash = checkpoints.get(step)
        cached = cache / f"{fingerprint}_{step}.json"
        if cached.exists():
            report = json.loads(cached.read_text())
        else:
            params = load_params(path, spec)
            report, geometry = analyze_checkpoint(
                spec, process, params, variant=variant, epsilon=epsilon,
                controls=step in control_steps, final=step == steps[-1], smoke=smoke)
            report.update(checkpoint_step=step, checkpoint_sha256=weight_hash,
                          archived_full_belief_error=archived_report["targets"]["belief"]["global_mse_ratio"])
            if geometry is not None:
                geometry_path = cache / f"simplex_{fingerprint}.npz"
                np.savez_compressed(geometry_path, **geometry)
                report["geometry"] = {"file": str(geometry_path.relative_to(HERE)),
                                      "sha256": sha256(geometry_path)}
            write_json(cached, report)
        reports.append(report)
        print(f"{arm} {len(reports)}/{len(steps)} step={step} "
              f"error={report['targets']['belief']['global_mse_ratio']:.6g} "
              f"elapsed={time.monotonic()-start:.1f}s", flush=True)
    import analysis
    output = {
        "schema_version": 1, "arm": arm, "exploration": exploration,
        "architecture": architecture, "variant": variant, "seed": 42,
        "protocol": protocol, "run_id": manifest["run_id"],
        "training_experiment_commit": manifest["git"]["commit"],
        "training_library_commit": manifest["git"]["library"]["commit"],
        "analysis_commit": revision(ROOT), "analysis_library_commit": revision(Path(analysis.__file__).parents[1]),
        "analysis_source_sha256": sha256(__file__), "source_manifest_sha256": manifest_hash,
        "archived_summary_sha256": summary_hash,
        "canonical_manifest_sha256": checkpoints.manifest_hash,
        "model": asdict(spec), "environment": environment.environment_metadata(),
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
        "devices": [str(d) for d in jax.devices()], "checkpoints": reports,
    }
    write_json(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--exploration", choices=STUDIES, nargs="+", default=list(STUDIES))
    parser.add_argument("--variant", choices=(1, 2, 3), type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Archive raw geometry and restart caches after successful analysis")
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    for architecture in args.architecture:
        for exploration in args.exploration:
            for variant in args.variant:
                print(run_arm(exploration, architecture, variant, args.smoke), flush=True)
    if args.upload:
        if args.smoke:
            raise ValueError("smoke artifacts are not campaign results")
        from harness.artifacts import finish_run_manifest, start_run_manifest, update_run_manifest
        from harness.context import RunContext
        from harness.storage.b2 import upload_run_artifacts

        context = RunContext(experiment_dir=HERE, results_dir=HERE / "results" / "paper_probes",
                             artifacts_dir=HERE / "artifacts" / "paper_probes",
                             run_id="paper-probes-pr219-20261114", seed=42)
        start_run_manifest(context, experiment_module=__name__, experiment_file=Path(__file__))
        summary = upload_run_artifacts(context, experiment_module=__name__)
        update_run_manifest(context, remote_artifacts=summary)
        finish_run_manifest(context, status="completed")
        print("Raw probe artifacts durably uploaded", flush=True)


if __name__ == "__main__":
    main()
