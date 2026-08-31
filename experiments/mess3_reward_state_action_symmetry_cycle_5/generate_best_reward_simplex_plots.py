"""Generate belief-simplex plots for the highest-reward checkpoint per variant."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))

import boto3
import numpy as np
import torch
from botocore.config import Config
from ray.rllib.algorithms.ppo import PPOConfig

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
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.analysis import (
    _install_checkpoint_import_aliases,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    _final_checkpoint_name,
)
from harness.seeding import named_seed_sequences

CYCLE_ROOT = _SCRIPT_DIR
# Historical B2 uploads used the pre-rename study slug.
B2_STUDY = "mess3_reward_state_action_asymmetry_cycle_5"
RETURN_METRICS = (
    "env_runners/episode_return_mean",
    "env_runners/agent_episode_return_mean/default_agent",
    "env_runners/module_episode_return_mean/default_policy",
)
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


def _b2_base(variant: int, run_id: str) -> str:
    return f"experiments/{B2_STUDY}/variant_{variant}/{run_id}"


def _download_json(s3, bucket: str, key: str) -> dict[str, Any]:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        return json.loads(body.read())
    finally:
        body.close()


def _metric(metrics: dict[str, Any], paths: tuple[str, ...]) -> float:
    for path in paths:
        value: Any = metrics
        missing = False
        for part in path.split("/"):
            if not isinstance(value, dict) or part not in value:
                missing = True
                break
            value = value[part]
        if not missing:
            return float(value)
    raise KeyError(f"missing metrics {paths!r}")


def _load_multi_seed_summary() -> dict[str, Any]:
    path = CYCLE_ROOT / "multi_seed_summary.json"
    return json.loads(path.read_text())


def _select_best_seed_by_reward(variant: int) -> tuple[int, str, float]:
    payload = _load_multi_seed_summary()
    arm = payload["arms"][f"variant_{variant}"]["per_seed"]
    best = max(arm, key=lambda item: float(item["mean_episode_return"]))
    return int(best["seed"]), str(best["run_id"]), float(best["mean_episode_return"])


def _download_checkpoints(
    s3,
    bucket: str,
    variant: int,
    run_id: str,
    checkpoint_name: str,
    artifacts_dir: Path,
) -> Path:
    base = _b2_base(variant, run_id)
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
    return best.resolve()


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


def _probe_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    device = _probe_device()

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
        "device": device,
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
            "device": device,
        },
        targets=test.beliefs[indices],
        predictions=predicted[indices],
    )


def generate_variant_plot(variant: int) -> dict[str, Any]:
    _install_checkpoint_import_aliases(cycle=5)
    seed, run_id, episode_return = _select_best_seed_by_reward(variant)
    s3 = _s3_client()
    bucket = os.environ["B2_BUCKET"]
    tune_key = f"{_b2_base(variant, run_id)}/compact-results/tune_summary.json"
    tune_summary = _download_json(s3, bucket, tune_key)
    checkpoint_name = _final_checkpoint_name(tune_summary)
    artifacts_dir = CYCLE_ROOT / f"variant_{variant}" / "artifacts" / run_id
    checkpoint = _download_checkpoints(
        s3,
        bucket,
        variant,
        run_id,
        checkpoint_name,
        artifacts_dir,
    )
    env_class, env_config = _load_env_config(checkpoint)
    agent_steps = int(
        _metric(
            tune_summary["trials"][0]["metrics"],
            ("num_env_steps_sampled_lifetime", "env_runners/num_env_steps_sampled_lifetime"),
        )
    )
    probe = _probe_module(
        seed=seed,
        checkpoint=checkpoint,
        agent_steps=agent_steps,
        env_class=env_class,
        env_config=env_config,
    )
    figures_dir = CYCLE_ROOT / "results" / "best_reward_simplex"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / f"variant_{variant}.png"
    plot_probe(
        probe,
        title=(
            f"variant_{variant} — best reward seed {seed}\n"
            f"return={episode_return:.1f}, steps={agent_steps:,}"
        ),
        path=output_path,
    )
    summary = {
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "checkpoint_name": checkpoint_name,
        "agent_steps": agent_steps,
        "episode_return_mean": episode_return,
        "reprobe_mse": float(probe.metrics["mse"]),
        "probe_device": probe.metrics["device"],
        "figure": str(output_path),
    }
    summary_path = figures_dir / f"variant_{variant}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    summaries = []
    for variant in args.variants:
        summary = generate_variant_plot(variant)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    combined = {
        "study": "mess3_reward_state_action_symmetry_cycle_5",
        "selection_metric": RETURN_METRICS[0],
        "variants": summaries,
    }
    output = CYCLE_ROOT / "results" / "best_reward_simplex" / "summaries.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2) + "\n")


if __name__ == "__main__":
    main()
