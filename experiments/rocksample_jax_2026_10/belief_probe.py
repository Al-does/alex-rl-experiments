"""Offline decision-time probing of the archived d128_kl0.1 agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import boto3
import jax
import jax.numpy as jnp
import numpy as np
from analysis.belief_geometry import evaluate_belief_geometry
from analysis.probes.controls import (
    fit_grouped_affine,
    paired_comparison,
    score_prediction,
)
from analysis.probes.resampling import cluster_bootstrap_statistics, percentile_interval
from envs.rocksample import (
    Action,
    Observation,
    configuration_bits,
    joint_from_marginals,
    update_marginals,
)

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm
from experiments.rocksample_jax_2026_10 import ppo

ROOT = Path(__file__).resolve().parent
RUNS = {5: "20261002T010437Z-3f1c4e88", 6: "20261002T050311Z-633fa674"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(seed: int) -> tuple[str, Path]:
    round_ = 5 if seed == 0 else 6
    relative = f"ppo_bmax_r{round_}/{RUNS[round_]}"
    return (
        f"experiments/rocksample_jax_2026_10/{relative}",
        ROOT / f"ppo_bmax_r{round_}/results/{RUNS[round_]}/resolved_recipe.json",
    )


def checkpoint(
    directory: Path,
    seed: int,
    download: bool,
    *,
    run_results: Path | None = None,
    update: int | None = None,
) -> tuple[dict, int, dict, dict]:
    if run_results is None:
        prefix, recipe_path = source(seed)
        manifest_path = directory / f"r{5 if seed == 0 else 6}_manifest.json"
        relative = f"d128_kl0.1/seed{seed}.pkl"
        expected_steps = None
    else:
        run_results = run_results.resolve()
        leaf = run_results.parent.parent
        prefix = f"{leaf.relative_to(ROOT.parents[1])}/{run_results.name}"
        recipe_path = run_results / "resolved_recipe.json"
        manifest_path = directory / "durability_manifest.json"
        records = checkpoint_records(run_results, seed)
        selected = next(r for r in records if r["update"] == update)
        relative, expected_steps = selected["path"], selected["env_steps"]
    recipe = json.loads(recipe_path.read_text())
    arm = next(a for a in recipe["arms"] if a["name"] == "d128_kl0.1")
    path = directory / relative if run_results else directory / f"seed{seed}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if download:
        client = boto3.client(
            "s3",
            endpoint_url=os.environ["B2_ENDPOINT"],
            aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
            aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        )
        if not manifest_path.exists():
            client.download_file(
                os.environ["B2_BUCKET"],
                f"{prefix}/metadata/durability_manifest.json",
                str(manifest_path),
            )
        if not path.exists():
            client.download_file(
                os.environ["B2_BUCKET"],
                f"{prefix}/{relative}",
                str(path),
            )
    manifest = json.loads(manifest_path.read_text())
    record = next(
        f
        for f in manifest["files"]
        if f["key"] == f"{prefix}/{relative}"
    )
    digest = sha256(path)
    if digest != record["sha256"] or path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"checkpoint failed durability verification: {path}")
    with path.open("rb") as stream:
        state: ppo.RunnerState = pickle.load(stream)
    params = jax.tree.map(jnp.asarray, state.params)
    steps = int(state.env_steps)
    if steps != (arm["env_steps_per_seed"] if expected_steps is None else expected_steps):
        raise ValueError("checkpoint steps disagree with recorded checkpoint metadata")
    return (
        params,
        steps,
        arm,
        {
            "uri": record["uri"],
            "sha256": digest,
            "size_bytes": record["size_bytes"],
            "runtime_seed": recipe["seed"],
            "num_keys": recipe.get("num_keys") or recipe["num_seeds"],
            "run_id": run_results.name if run_results else prefix.rsplit("/", 1)[1],
            "update": update,
            "recipe_sha256": sha256(recipe_path),
            "durability_manifest_sha256": sha256(manifest_path),
        },
    )


def checkpoint_records(run_results: Path, seed: int) -> list[dict]:
    summary = json.loads((run_results / "summary.json").read_text())
    selected = next(
        s for s in summary["arms"]["d128_kl0.1"]["seeds"] if s["seed_index"] == seed
    )
    records = sorted(selected["checkpoints"], key=lambda r: r["update"])
    if not records or any(
        a["update"] >= b["update"] or a["env_steps"] >= b["env_steps"]
        for a, b in zip(records, records[1:])
    ):
        raise ValueError("checkpoint updates and steps must increase strictly")
    for record in records:
        path = Path(record["path"])
        if path.is_absolute() or ".." in path.parts or not (
            path == Path(f"d128_kl0.1/seed{seed}.pkl")
            or path.parent == Path(f"d128_kl0.1/seed{seed}")
        ):
            raise ValueError("checkpoint path disagrees with seed or artifact scope")
    return records


def initialization(
    spec: tm.TransformerSpec, seed: int, runtime_seed: int, num_keys: int = 4
) -> dict:
    run_key = jax.random.split(jax.random.key(runtime_seed), num_keys)[seed]
    _, param_key, _ = jax.random.split(run_key, 3)
    return tm.init_params(spec, param_key)


def check_route(env: rs.RockSampleParams) -> np.ndarray:
    actions = list(range(5, 5 + env.k))
    x, y = env.start
    for i, (rx, ry) in enumerate(env.rocks):
        actions += [int(Action.EAST if rx > x else Action.WEST)] * abs(rx - x)
        actions += [int(Action.NORTH if ry > y else Action.SOUTH)] * abs(ry - y)
        actions += [5 + i, int(Action.SAMPLE)]
        x, y = rx, ry
    actions += [int(Action.EAST)] * (env.n - x)
    if len(actions) > env.episode_length:
        raise ValueError("diagnostic route exceeds the episode cap")
    return np.pad(
        np.asarray(actions, np.int32),
        (0, env.episode_length - len(actions)),
        constant_values=int(Action.EAST),
    )


@partial(jax.jit, static_argnames=("spec", "env", "episodes", "key_seed", "scripted"))
def collect_device(
    spec: tm.TransformerSpec,
    env: rs.RockSampleParams,
    params: dict,
    init_params: dict,
    *,
    episodes: int,
    key_seed: int,
    scripted: bool = False,
) -> dict[str, jax.Array]:
    """Complete episodes, both encoders seeing exactly the same input histories."""
    obs, state = jax.vmap(partial(rs.reset, env))(
        jax.random.split(jax.random.key(key_seed), episodes)
    )
    cache = jax.vmap(lambda _: tm.empty_cache(spec))(jnp.arange(episodes))
    route = jnp.asarray(check_route(env))

    def advance(carry, t):
        obs, state, final_cache, init_cache, done, key = carry
        z, final_cache = jax.vmap(partial(tm.encode_step, spec, params))(
            final_cache, obs
        )
        z0, init_cache = jax.vmap(partial(tm.encode_step, spec, init_params))(
            init_cache, obs
        )
        logits = z @ params["policy"]["w"] + params["policy"]["b"]
        key, action_key, env_key = jax.random.split(key, 3)
        action = (
            jnp.full((episodes,), route[t])
            if scripted
            else jax.random.categorical(action_key, logits).astype(jnp.int32)
        )
        out = jax.vmap(partial(rs.step, env))(
            jax.random.split(env_key, episodes), state, action
        )
        row = {
            "features": z,
            "initialization": z0,
            "obs": obs,
            "position": state.rover,
            "qualities": state.qualities,
            "action": action,
            "symbol": out.state.symbol,
            "reward": out.reward,
            "valid": ~done,
            "terminated": out.terminated,
            "truncated": out.truncated,
        }
        done |= out.terminated | out.truncated
        return (out.obs, out.state, final_cache, init_cache, done, key), row

    carry = (
        obs,
        state,
        cache,
        cache,
        jnp.zeros(episodes, bool),
        jax.random.key(key_seed + 1),
    )
    _, rows = jax.lax.scan(advance, carry, jnp.arange(env.episode_length))
    return {name: jnp.swapaxes(value, 0, 1) for name, value in rows.items()}


def collect(
    spec: tm.TransformerSpec,
    env: rs.RockSampleParams,
    params: dict,
    init_params: dict,
    *,
    episodes: int,
    key_seed: int,
    scripted: bool = False,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value)
        for name, value in collect_device(
            spec, env, params, init_params,
            episodes=episodes, key_seed=key_seed, scripted=scripted,
        ).items()
    }


@dataclass
class Dataset:
    features: np.ndarray
    initialization: np.ndarray
    marginals: np.ndarray
    joint: np.ndarray
    nuisance: np.ndarray
    predictive: np.ndarray
    relevant: np.ndarray
    groups: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    checks: np.ndarray
    samples: np.ndarray
    good_samples: np.ndarray
    final_belief: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    exits: np.ndarray
    calibration: dict


def targets(rows: dict[str, np.ndarray], env: rs.RockSampleParams) -> Dataset:
    n, horizon = rows["valid"].shape
    beliefs = np.zeros((n, horizon, env.k))
    relevant = np.zeros_like(beliefs, bool)
    sampled_history = np.zeros_like(beliefs, bool)
    checks = np.zeros((n, env.k), int)
    samples = np.zeros_like(checks)
    good_samples = np.zeros_like(checks)
    final = np.zeros((n, env.k))
    bins = np.zeros((10, 3))
    sensor = np.zeros(3)
    rocks = np.asarray(env.rocks)
    for episode in range(n):
        m = np.full(env.k, 0.5)
        sampled = np.zeros(env.k, bool)
        for t in np.flatnonzero(rows["valid"][episode]):
            beliefs[episode, t] = m
            relevant[episode, t] = ~sampled & (m > 0)
            sampled_history[episode, t] = sampled
            for rock in np.flatnonzero(~sampled):
                b = min(int(m[rock] * 10), 9)
                bins[b] += [1, m[rock], rows["qualities"][episode, t, rock]]
            action = int(rows["action"][episode, t])
            position = rows["position"][episode, t]
            symbol = int(rows["symbol"][episode, t])
            if action >= 5:
                rock = action - 5
                checks[episode, rock] += 1
                eta = 2 ** (
                    -np.linalg.norm(position - rocks[rock])
                    / env.half_efficiency_distance
                )
                sensor += [1, (1 - eta) / 2 + m[rock] * eta, symbol == Observation.GOOD]
            if action == Action.SAMPLE:
                at_rock = np.all(rocks == position, axis=1)
                samples[episode] += at_rock
                good_samples[episode] += at_rock & (rows["reward"][episode, t] > 0)
                sampled |= at_rock
            if not rows["terminated"][episode, t]:
                m = update_marginals(
                    m,
                    rocks=rocks,
                    position=position,
                    action=action,
                    symbol=symbol,
                    half_efficiency_distance=env.half_efficiency_distance,
                )
        final[episode] = m
    mask = rows["valid"]
    groups, times = np.nonzero(mask)
    m = beliefs[mask]
    position = rows["position"][mask]
    eta = 2 ** (
        -np.linalg.norm(position[:, None, :] - rocks, axis=-1)
        / env.half_efficiency_distance
    )
    predictive = (1 - eta) / 2 + m * eta
    nuisance = np.column_stack(
        (rows["obs"][mask], times / horizon, sampled_history[mask])
    )
    return Dataset(
        rows["features"][mask],
        rows["initialization"][mask],
        m,
        joint_from_marginals(m),
        nuisance,
        predictive,
        relevant[mask],
        groups,
        times,
        position,
        rows["action"][mask],
        rows["reward"][mask],
        checks,
        samples,
        good_samples,
        final,
        (rows["reward"] * mask).sum(axis=1),
        mask.sum(axis=1),
        (rows["terminated"] & mask).any(axis=1),
        {
            "bins_count_sum_expected_sum_true": bins.tolist(),
            "sensor_count_sum_expected_sum_good": sensor.tolist(),
        },
    )


def split(data: Dataset) -> tuple[np.ndarray, np.ndarray]:
    groups = np.unique(data.groups)
    fit_groups = np.random.default_rng(831).permutation(groups)[: len(groups) // 2]
    fit = np.isin(data.groups, fit_groups)
    return fit, ~fit


def metrics(pred: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict:
    def statistic(indices: np.ndarray) -> float:
        ratio = score_prediction(pred[indices], target[indices])["normalized_mse"]
        return np.nan if ratio is None else ratio

    values = cluster_bootstrap_statistics(groups, statistic, seed=551, n_resamples=200)
    return {
        **score_prediction(pred, target),
        "normalized_mse_ci": list(percentile_interval(values))
        if np.isfinite(values).all()
        else None,
        "bootstrap_unit": "episode",
        "bootstrap_refit": False,
    }


def analyze(
    data: Dataset, *, joint_battery: bool = True
) -> tuple[dict, dict[str, np.ndarray]]:
    fit, test = split(data)
    y = data.marginals
    features = {
        "final": data.features,
        "initialization": data.initialization,
        "observable_nuisance": data.nuisance,
        "next_sensor": data.predictive,
    }
    predictions = {}
    report = {
        "n_fit_episodes": len(np.unique(data.groups[fit])),
        "n_test_episodes": len(np.unique(data.groups[test])),
        "marginals": {},
    }
    for name, x in features.items():
        weight, bias, cv = fit_grouped_affine(
            x[fit], y[fit], data.groups[fit], seed=551
        )
        prediction = x[test] @ weight + bias
        predictions[name] = prediction
        per_rock = []
        for rock in range(y.shape[1]):
            relevant = data.relevant[test, rock]
            informed = relevant & (y[test, rock] != 0.5)
            masks = {
                "all": np.ones(test.sum(), bool),
                "relevant": relevant,
                "informed_relevant": informed,
            }
            per_rock.append(
                {
                    label: score_prediction(prediction[mask, rock], y[test][mask, rock])
                    if mask.any()
                    else None
                    for label, mask in masks.items()
                }
            )
        report["marginals"][name] = {
            "metrics": metrics(prediction, y[test], data.groups[test]),
            "per_rock": per_rock,
            "fit": cv,
        }
    report["marginals"]["final_vs_initialization"] = paired_comparison(
        predictions["final"],
        predictions["initialization"],
        y[test],
        data.groups[test],
        seed=551,
        n_resamples=200,
    )
    if joint_battery:
        bits = configuration_bits(y.shape[1]).astype(float)
        battery = evaluate_belief_geometry(
            {"final": data.features[fit]},
            {"final": data.features[test]},
            data.joint[fit],
            data.joint[test],
            train_groups=data.groups[fit],
            test_groups=data.groups[test],
            initialization_features={
                "final": (data.initialization[fit], data.initialization[test])
            },
            nuisance_features={
                "observables": (data.nuisance[fit], data.nuisance[test]),
                "next_sensor": (data.predictive[fit], data.predictive[test]),
            },
            contrasts={f"rock{i}_good": bits[:, i] for i in range(y.shape[1])},
            seed=551,
            n_null_repeats=3,
            n_resamples=200,
        )
        report["joint"] = battery.report
    else:
        report["joint"] = {}
        for name in ("final", "initialization"):
            x = features[name]
            w, b, cv = fit_grouped_affine(
                x[fit], data.joint[fit], data.groups[fit], seed=551
            )
            report["joint"][name] = {
                "metrics": score_prediction(x[test] @ w + b, data.joint[test]),
                "fit": cv,
            }
    report["behavior"] = behavior(data)
    report["validation"] = {
        "joint_sum_max_error": float(np.abs(data.joint.sum(axis=1) - 1).max()),
        "marginal_joint_max_error": float(
            np.abs(data.joint @ configuration_bits(y.shape[1]) - y).max()
        ),
        "all_reset_targets_half": bool(np.all(y[data.times == 0] == 0.5)),
        "calibration": data.calibration,
    }
    test_rows = np.flatnonzero(test)
    example = data.groups[test_rows[0]]
    selected = data.groups[test_rows] == example
    report["example"] = {
        "episode": int(example),
        "times": data.times[test][selected].tolist(),
        "targets": y[test][selected].tolist(),
        "predictions": predictions["final"][selected].tolist(),
        "actions": data.actions[test][selected].tolist(),
    }
    report["check_updates"] = []
    index = {
        (int(g), int(t)): i
        for i, (g, t) in enumerate(
            zip(data.groups[test], data.times[test], strict=True)
        )
    }
    for rock in range(y.shape[1]):
        before, after = [], []
        for (group, time), i in index.items():
            if data.actions[test][i] == 5 + rock and (group, time + 1) in index:
                before.append(i)
                after.append(index[(group, time + 1)])
        change = y[test][after, rock] - y[test][before, rock]
        decoded_change = (
            predictions["final"][after, rock] - predictions["final"][before, rock]
        )
        informative = np.abs(change) > 1e-8
        report["check_updates"].append(
            {
                "n_checks": len(before),
                "n_informative": int(informative.sum()),
                "metrics": score_prediction(
                    decoded_change[informative], change[informative]
                )
                if informative.any()
                else None,
                "mean_absolute_target_change": float(np.abs(change).mean())
                if len(change)
                else None,
                "mean_absolute_decoded_change": float(np.abs(decoded_change).mean())
                if len(change)
                else None,
            }
        )
    return report, predictions


def behavior(data: Dataset) -> dict:
    return {
        "episodes": len(data.returns),
        "return_mean": float(data.returns.mean()),
        "return_sem": float(data.returns.std(ddof=1) / np.sqrt(len(data.returns))),
        "returns": data.returns.tolist(),
        "length_mean": float(data.lengths.mean()),
        "exit_fraction": float(data.exits.mean()),
        "checks_per_episode": data.checks.mean(axis=0).tolist(),
        "samples_per_episode": data.samples.mean(axis=0).tolist(),
        "check_episode_fraction": (data.checks > 0).mean(axis=0).tolist(),
        "sample_episode_fraction": (data.samples > 0).mean(axis=0).tolist(),
        "good_sample_episode_fraction": (data.good_samples > 0).mean(axis=0).tolist(),
        "prior_at_end_fraction": (data.final_belief == 0.5).mean(axis=0).tolist(),
    }


def revision(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "artifacts/belief_checkpoints"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", choices=range(4), default=list(range(4))
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = rs.RockSampleParams.from_instance()
    episodes = 16 if args.smoke else 512
    for seed in args.seeds:
        print(f"seed {seed}: verify and load checkpoint", flush=True)
        params, steps, arm, provenance = checkpoint(
            args.checkpoint_dir, seed, args.download
        )
        spec = tm.TransformerSpec(
            env.obs_dim,
            env.num_actions,
            arm["d_model"],
            arm["n_layers"],
            arm["n_heads"],
            arm["context_len"],
        )
        init_params = initialization(spec, seed, provenance["runtime_seed"])
        if (
            params["input"]["w"].shape != (env.obs_dim, spec.d_model)
            or len(params["blocks"]) != spec.n_layers
        ):
            raise ValueError("checkpoint architecture disagrees with recipe")
        report = {
            "seed": seed,
            "env_steps": steps,
            "source": provenance,
            "spec": asdict(spec),
            "env": asdict(env),
            "smoke": args.smoke,
            "episodes_per_distribution": episodes,
            "jax_version": jax.__version__,
            "experiment_revision": revision(ROOT),
            "harness_revision": revision(ROOT.parents[2] / "rl-harness"),
            "analysis_source_sha256": sha256(Path(__file__)),
            "initialization": "Reconstructed exact sweep split(key(42),4)[seed], then ppo.init parameter-key split; no optimization",
            "sampling": "Complete reset-to-exit/cap episodes, no warmup removal; fit/test episode split seed 831; train-only SVD cutoff CV; unprojected affine predictions",
        }
        for name, model, scripted in [
            ("on_policy", params, False),
            ("common_checks", params, True),
            ("initial_policy", init_params, False),
        ]:
            print(f"seed {seed}: collect/analyze {name}", flush=True)
            data = targets(
                collect(
                    spec,
                    env,
                    model,
                    init_params,
                    episodes=episodes,
                    key_seed=9131,
                    scripted=scripted,
                ),
                env,
            )
            report[name], _ = analyze(
                data, joint_battery=name == "on_policy" and not args.smoke
            )
        path = args.output / f"seed{seed}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"seed {seed}: saved {path}", flush=True)


if __name__ == "__main__":
    main()
