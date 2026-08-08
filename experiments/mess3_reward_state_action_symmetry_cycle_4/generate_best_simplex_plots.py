"""Generate init-vs-best belief simplex plots for cycle-4 PPO variants."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))

import boto3
import matplotlib.pyplot as plt
import numpy as np
from botocore.config import Config
from ray.rllib.algorithms.ppo import PPOConfig

from analysis.plots import simplex_scatter
from analysis.probes import fit_affine_probe, mean_squared_error, probe_predict
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_2.analysis import (
    N_ENVS,
    PROBE_RIDGE,
    ProbeResult,
    plot_probe,
)
from harness.seeding import named_seed_sequences

CYCLE_ROOT = _SCRIPT_DIR
SEEDS = (42, 43, 44, 45, 46)
_STREAM_KEYS = {
    "probe_train": (300,),
    "probe_test": (301,),
    "plot_sample": (305,),
}


def _s3_client():
    endpoint = os.environ["B2_ENDPOINT"].rstrip("/")
    if not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=Config(signature_version="s3v4"),
    )


def _sync_prefix(s3, bucket: str, prefix: str, dest_root: Path) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == obj["Size"]:
                continue
            s3.download_file(bucket, key, str(target))


def _download_curve(s3, bucket: str, variant: int, seed: int, out_dir: Path) -> dict:
    run_id = f"mess3-rsa-c4-v{variant}-seed{seed}"
    key = (
        "experiments/mess3_reward_state_action_symmetry_cycle_4/"
        f"variant_{variant}/{run_id}/compact-results/checkpoint_probe_curve.json"
    )
    dest = out_dir / run_id / "checkpoint_probe_curve.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest))
    return json.loads(dest.read_text())


def _best_trained_checkpoint(curve: dict) -> dict:
    trained = [
        point
        for point in curve["checkpoints"]
        if int(point.get("agent_steps") or 0) > 0
    ]
    if not trained:
        raise ValueError("checkpoint probe curve has no trained checkpoints")
    return min(trained, key=lambda point: float(point["mse"]))


def _select_best_seed(variant: int, results_dir: Path) -> tuple[int, dict, dict]:
    best_seed = None
    best_point = None
    best_curve = None
    for seed in SEEDS:
        curve = _download_curve(_s3_client(), os.environ["B2_BUCKET"], variant, seed, results_dir)
        point = _best_trained_checkpoint(curve)
        if best_point is None or float(point["mse"]) < float(best_point["mse"]):
            best_seed = seed
            best_point = point
            best_curve = curve
    if best_seed is None or best_point is None or best_curve is None:
        raise RuntimeError(f"variant {variant} has no probe curves")
    return best_seed, best_point, best_curve


def _download_checkpoints(
    s3,
    bucket: str,
    variant: int,
    run_id: str,
    checkpoint_name: str,
    artifacts_dir: Path,
) -> tuple[Path, Path]:
    base = (
        "experiments/mess3_reward_state_action_symmetry_cycle_4/"
        f"variant_{variant}/{run_id}"
    )
    initial = artifacts_dir / "initial_checkpoint"
    _sync_prefix(s3, bucket, f"{base}/initial_checkpoint/", initial)
    best_prefix = None
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{base}/tune/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if f"/{checkpoint_name}/rllib_checkpoint.json" in key:
                best_prefix = key[: key.index(f"/{checkpoint_name}/") + len(checkpoint_name) + 1]
                break
        if best_prefix:
            break
    if best_prefix is None:
        raise FileNotFoundError(f"missing {checkpoint_name} for {run_id}")
    rel = best_prefix[len(f"{base}/") :]
    best = artifacts_dir / rel
    _sync_prefix(s3, bucket, best_prefix, best)
    return initial.resolve(), best.resolve()


def _load_env_config(reference_checkpoint: Path):
    with open(reference_checkpoint / "class_and_ctor_args.pkl", "rb") as handle:
        spec = pickle.load(handle)
    _, (args, kwargs) = spec["class"], spec["ctor_args_and_kwargs"]
    config = PPOConfig().from_dict(kwargs.get("config") or args[0])
    env_config = dict(config.env_config)
    env_config["diagnostics"] = {
        "state": True,
        "belief": True,
        "tokens": True,
        "transitions": True,
    }
    return config.env, env_config


def _load_module(checkpoint: Path):
    policy_dir = checkpoint / "learner_group/learner/rl_module/default_policy"
    with open(policy_dir / "class_and_ctor_args.pkl", "rb") as handle:
        spec = pickle.load(handle)
    cls = spec["class"]
    args, kwargs = spec["ctor_args_and_kwargs"]
    module = cls(*args, **kwargs)
    with open(policy_dir / "module_state.pkl", "rb") as handle:
        state = pickle.load(handle)
    if hasattr(module, "set_state"):
        module.set_state(state)
    else:
        module.load_state_dict(state)
    return module


def _probe_module(
    *,
    seed: int,
    checkpoint: Path,
    agent_steps: int,
    env_class,
    env_config,
) -> ProbeResult:
    streams = named_seed_sequences(seed, _STREAM_KEYS)
    module = _load_module(checkpoint)

    def make_environment():
        return env_class(env_config)

    environment = make_environment()
    try:
        transducer_target = make_transducer_target(environment)
    finally:
        environment.close()
    common = {
        "module": module,
        "env_factory": make_environment,
        "policy_mode": "greedy",
        "device": "cpu",
        "warmup": 64,
        "n_envs": N_ENVS,
        "initial_belief": transducer_target[0],
        "action_outcome_operator": transducer_target[1],
        "initial_outcome_operator": transducer_target[2],
    }
    train = collect_probe_data(
        n_steps=60_000,
        seed=streams["probe_train"],
        **common,
    )
    test = collect_probe_data(
        n_steps=80_000,
        seed=streams["probe_test"],
        **common,
    )
    weight, bias = fit_affine_probe(
        train.activations,
        train.beliefs,
        ridge=PROBE_RIDGE,
    )
    predicted = probe_predict(weight, bias, test.activations)
    sample_size = min(20_000, len(test.beliefs))
    rng = np.random.default_rng(streams["plot_sample"])
    indices = rng.choice(len(test.beliefs), sample_size, replace=False)
    return ProbeResult(
        metrics={
            "mse": float(mean_squared_error(predicted, test.beliefs)),
            "checkpoint_step": agent_steps,
        },
        targets=test.beliefs[indices],
        predictions=predicted[indices],
    )


def _display(result: ProbeResult) -> np.ndarray:
    display = np.clip(result.predictions, 0.0, None)
    display /= np.maximum(display.sum(axis=1, keepdims=True), 1e-12)
    return display


def _plot_init_vs_best(
    variant: int,
    initial: ProbeResult,
    best: ProbeResult,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 5.8), constrained_layout=True)
    colors_init = np.clip(initial.targets, 0.0, 1.0)
    colors_best = np.clip(best.targets, 0.0, 1.0)
    panels = [
        (initial.targets, colors_init, f"variant_{variant} — init: exact belief"),
        (
            _display(initial),
            colors_init,
            (
                f"variant_{variant} — init: affine decoded\n"
                f"MSE={initial.metrics['mse']:.5f}"
            ),
        ),
        (best.targets, colors_best, f"variant_{variant} — best: exact belief"),
        (
            _display(best),
            colors_best,
            (
                f"variant_{variant} — best: affine decoded\n"
                f"MSE={best.metrics['mse']:.5f}"
            ),
        ),
    ]
    for axis, (points, colors, title) in zip(axes, panels):
        simplex_scatter(
            axis,
            points,
            colors=colors,
            s=0.3,
            alpha=0.2,
            title=title,
            labels=("s0", "s1", "s2"),
        )
        axis.set_title(title, fontsize=10, pad=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def generate_variant_plot(variant: int) -> dict:
    variant_dir = CYCLE_ROOT / f"variant_{variant}"
    results_dir = variant_dir / "results"
    seed, best_point, _curve = _select_best_seed(variant, results_dir)
    run_id = f"mess3-rsa-c4-v{variant}-seed{seed}"
    artifacts_dir = variant_dir / "artifacts" / run_id
    run_results = results_dir / run_id
    s3 = _s3_client()
    bucket = os.environ["B2_BUCKET"]
    initial_ckpt, best_ckpt = _download_checkpoints(
        s3,
        bucket,
        variant,
        run_id,
        str(best_point["checkpoint_name"]),
        artifacts_dir,
    )
    env_class, env_config = _load_env_config(initial_ckpt)
    initial_probe = _probe_module(
        seed=seed,
        checkpoint=initial_ckpt,
        agent_steps=0,
        env_class=env_class,
        env_config=env_config,
    )
    best_probe = _probe_module(
        seed=seed,
        checkpoint=best_ckpt,
        agent_steps=int(best_point["agent_steps"]),
        env_class=env_class,
        env_config=env_config,
    )
    plot_probe(
        initial_probe,
        title=f"variant_{variant} — init",
        path=run_results / "belief_simplex_init.png",
    )
    plot_probe(
        best_probe,
        title=f"variant_{variant} — best",
        path=run_results / "belief_simplex_best.png",
    )
    comparison = run_results / "belief_simplex_init_vs_best.png"
    _plot_init_vs_best(variant, initial_probe, best_probe, comparison)
    summary = {
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "best_checkpoint": best_point["checkpoint_name"],
        "best_agent_steps": int(best_point["agent_steps"]),
        "best_training_iteration": best_point.get("training_iteration"),
        "curve_mse": float(best_point["mse"]),
        "reprobe_mse": float(best_probe.metrics["mse"]),
        "initial_mse": float(initial_probe.metrics["mse"]),
        "figures": {
            "init": str(run_results / "belief_simplex_init.png"),
            "best": str(run_results / "belief_simplex_best.png"),
            "init_vs_best": str(comparison),
        },
    }
    (run_results / "best_simplex_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    for variant in args.variants:
        summary = generate_variant_plot(variant)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
