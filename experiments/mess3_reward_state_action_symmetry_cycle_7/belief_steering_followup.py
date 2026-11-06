"""Offline Full-State Quotient residual steering; no training or Ray startup."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_prediction import (
    BeliefPolicy, LAMBDAS, prediction_curve, simulate, target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import (
    WorkItem, _download_module,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import environment_config

STUDY = Path(__file__).resolve().parent
RESULTS = STUDY / "variant_3_ctx32_ent/results"
ARTIFACTS = STUDY / "variant_3_ctx32_ent/artifacts/belief_steering"
SITES = ("block_1", "block_2", "block_3", "block_4", "post_final_norm")


@dataclass
class BeliefMap:
    decoder: np.ndarray
    bias: np.ndarray
    embedding: np.ndarray


def affine(x, y):
    fit = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    return fit[:-1], fit[-1]


def fit_map(x, b):
    decoder, bias = affine(x, b[:, :2])
    embedding, _ = affine(b[:, :2], x)
    return BeliefMap(decoder, bias, embedding)


def shift(x, mapping, mode):
    decoded = x @ mapping.decoder + mapping.bias
    b = np.column_stack((decoded, 1-decoded.sum(axis=1)))
    return (target(b, mode)[:, :2]-decoded) @ mapping.embedding


def orthogonal_direction(mapping, seed):
    direction = np.random.default_rng(seed).standard_normal(mapping.decoder.shape[0])
    direction -= (direction @ mapping.decoder) @ np.linalg.pinv(mapping.decoder, rcond=1e-10)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        raise ValueError("empty decoder null space")
    return direction/norm


class ResidualSites:
    """MLP-output hook shifts the post-block residual, before the next block."""

    def __init__(self, module):
        self.module = module
        self.before = {}
        self.activations = {}
        self.handles = []
        self.site = None
        self.mapping = None
        self.strength = 0.
        self.mode = "point"
        self.direction = None

    def __enter__(self):
        for layer, block in enumerate(self.module.encoder.blocks):
            def before(_module, inputs, _output, index=layer):
                self.before[index] = inputs[0]

            def after(_module, _inputs, output, index=layer):
                self.activations[index] = (self.before[index]+output)[:, -1, :]
                if self.site != index:
                    return None
                delta = self.delta(self.activations[index])
                return output + delta[:, None, :]

            self.handles.append(block.ln2.register_forward_hook(before))
            self.handles.append(block.mlp.register_forward_hook(after))
        return self

    def delta(self, activation):
        delta = shift(activation.cpu().numpy(), self.mapping, self.mode)
        if self.direction is not None:
            delta = np.linalg.norm(delta, axis=1, keepdims=True)*self.direction
        return torch.as_tensor(self.strength*delta, dtype=activation.dtype)

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


def config(horizon):
    cfg = environment_config(3)
    cfg.update(episode_length=horizon, randomize_first_episode_length=False,
               diagnostics={"belief": True, "state": True})
    return cfg


@torch.inference_mode()
def collect_sites(module, steps, seed, horizon=1024):
    if steps < 1 or horizon <= 8:
        raise ValueError("positive retained steps and horizon > warmup required")
    xs, bs = [], []
    n = 8
    envs = [HMMEnv(config(horizon)) for _ in range(n)]
    try:
        reset = [env.reset(seed=int(s)) for env, s in zip(
            envs, np.random.default_rng(seed).integers(2**31, size=n))]
        obs, infos = map(list, zip(*reset))
        state = _initial_state(module, n, torch.device("cpu"))
        retained = 0
        with ResidualSites(module) as sites:
            t = 0
            while retained < steps:
                h, state = module.encode_step(torch.as_tensor(np.stack(obs)), state)
                actions = module.action_distribution_inputs(h).argmax(-1).numpy()
                if t >= 8:
                    xs.append(torch.stack([sites.activations[i] for i in range(4)]+[h], 1).numpy())
                    bs.append(np.stack([info["belief_current"] for info in infos]))
                    retained += n
                outcomes = [env.step(int(a)) for env, a in zip(envs, actions)]
                obs = [v[0] for v in outcomes]
                infos = [v[4] for v in outcomes]
                t += 1
                if t == horizon:
                    reset = [env.reset() for env in envs]
                    obs, infos = map(list, zip(*reset))
                    state = _initial_state(module, n, torch.device("cpu"))
                    t = 0
        return np.concatenate(xs)[:steps].astype(np.float64), np.concatenate(bs)[:steps]
    finally:
        for env in envs:
            env.close()


@torch.inference_mode()
def evaluate(module, mapping, policy, *, site=3, strength=0., mode="point",
             episodes=32, seed=2042, horizon=1024, random_actions=False, matched_null=False):
    if episodes < 2 or horizon < 1:
        raise ValueError("at least two complete episodes required")
    rng = np.random.default_rng(seed)
    env_seeds = rng.integers(0, 2**31, episodes)
    action_seeds = rng.integers(0, 2**31, episodes)
    occupied = np.zeros(episodes)
    counts = np.zeros((3, 3), dtype=int)
    changed = agreement = total = 0
    for start in range(0, episodes, 16):
        n = min(16, episodes-start)
        envs = [HMMEnv(config(horizon)) for _ in range(n)]
        try:
            reset = [env.reset(seed=int(s)) for env, s in zip(envs, env_seeds[start:start+n])]
            obs, infos = map(list, zip(*reset))
            generators = [np.random.default_rng(int(s)) for s in action_seeds[start:start+n]]
            state = _initial_state(module, n, torch.device("cpu"))
            with ResidualSites(module) as sites:
                sites.mapping, sites.mode, sites.strength = mapping, mode, strength
                sites.direction = orthogonal_direction(mapping, seed+9999) if matched_null else None
                for step in range(horizon):
                    tensor = torch.as_tensor(np.stack(obs))
                    sites.site = None
                    # encode_step uses newly concatenated caches, not in-place writes.
                    shadow, shadow_state = module.encode_step(tensor, state)
                    base_actions = module.action_distribution_inputs(shadow).argmax(-1).numpy()
                    if random_actions:
                        actions = np.array([g.integers(3) for g in generators])
                        state = shadow_state
                    else:
                        sites.site = site if site < 4 else None
                        h, state = module.encode_step(tensor, state)
                        if site == 4:
                            h = h + sites.delta(h)
                        actions = module.action_distribution_inputs(h).argmax(-1).numpy()
                    b = np.stack([info["belief_current"] for info in infos])
                    installed = (1-strength)*b+strength*target(b, mode)
                    agreement += int(np.sum(actions == policy(installed)))
                    changed += int(np.sum(actions != base_actions))
                    latent = np.array([info["state_current"] for info in infos])
                    occupied[start:start+n] += latent == 2
                    np.add.at(counts, (latent, actions), 1)
                    total += n
                    outcomes = [env.step(int(a)) for env, a in zip(envs, actions)]
                    if any((v[2] or v[3]) != (step == horizon-1) for v in outcomes):
                        raise ValueError("evaluation episode horizon mismatch")
                    obs, infos = [v[0] for v in outcomes], [v[4] for v in outcomes]
        finally:
            for env in envs:
                env.close()
    occupied /= horizon
    conditional = [[float(c/row.sum()) for c in row] if row.sum() else [None]*3 for row in counts]
    return {"occupancy": float(occupied.mean()),
            "occupancy_se": float(occupied.std(ddof=1)/np.sqrt(episodes)),
            "action_fractions": (counts.sum(0)/total).tolist(),
            "action_change_rate": changed/total,
            "action_given_state": conditional, "state_action_counts": counts.tolist(),
            "noop_given_reward_state": conditional[2][0],
            "installed_belief_policy_agreement": agreement/total,
            "steps": total, "episodes": episodes}


def select_lambda(sweep, random_occupancy):
    eligible = [l for l in LAMBDAS if sweep[str(l)]["occupancy"] <= random_occupancy]
    return min(eligible) if eligible else min(LAMBDAS, key=lambda l: sweep[str(l)]["occupancy"])


def restore(seed):
    run_id = f"mess3_reward_state_action_symmetry_cycle_6-variant_3_ctx32_ent-seed{seed}-10m"
    run_dir = RESULTS/run_id
    curve = run_dir/"checkpoint_probe_curve.json"
    record = json.loads(curve.read_text())["checkpoints"][-1]
    path = ARTIFACTS/run_id/"default_policy"
    item = WorkItem("variant_3_ctx32_ent", 3, seed, "final", 6,
                    record.get("training_iteration"), int(record["agent_steps"]), run_id,
                    record["checkpoint_name"], curve, run_dir/"run_manifest.json", path,
                    ARTIFACTS/"unused.json")
    return path, {"run_id": run_id, "agent_steps": item.agent_steps,
                  "training_manifest": json.loads(item.run_manifest_path.read_text()),
                  **_download_module(item)}


def revision(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def write_report(output, report):
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# Full-State Quotient belief steering", "",
             "Occupancy is decision-time reward-state occupancy, including reset decisions.",
             "Actions: Noop, Tilt X, Tilt Y. States: 1, 2, 3 (reward state).", "",
             "Reference: N=60 nearest-grid, gamma=0.97, 800 iterations; near-optimal, not certified Bayes-optimal.",
             "Raw affine decoded beliefs are not clipped. Point ties choose state 2.",
             "Warmup is eight steps only for probe fitting/testing (baseline protocol); the banded four-layer context-32 encoder can depend on up to 128 observations.",
             "Shadow forward uses the current executed cache and is discarded; earlier shifts affect subsequent-block caches only.",
             "Selected strengths: "+json.dumps(report.get("selected", {})), "",
             "Uncertainty: "+report["uncertainty"], "",
             "| Condition | Occupancy ± SE |", "|---|---:|"]
    for name, result in report["conditions"].items():
        lines.append(f"| {name} | {result['occupancy']:.6f} ± {result['occupancy_se']:.6f} |")
    lines += ["", "## Final pre-LayerNorm curves", "",
              "| λ | Point ± SE | Mirror ± SE | Predicted point | Predicted mirror |",
              "|---:|---:|---:|---:|---:|"]
    for l in LAMBDAS:
        values = [report["curves"][m][str(l)] for m in ("point", "mirror")]
        predicted = [report["prediction"][m][str(l)]["occupancy"] for m in ("point", "mirror")]
        lines.append(f"| {l:g} | "+" | ".join(f"{v['occupancy']:.6f} ± {v['occupancy_se']:.6f}" for v in values)+f" | {predicted[0]:.6f} | {predicted[1]:.6f} |")
    output.with_suffix(".md").write_text("\n".join(lines)+"\n")


def run_seed(seed, policy, prediction, *, steps, tune_episodes, confirm_episodes, horizon, output):
    started = time.monotonic()
    checkpoint, provenance = restore(seed)
    module = load_module_only(checkpoint).to("cpu").eval()
    if len(module.encoder.blocks) != 4 or module.model_config["d_model"] != 64 or module.model_config["context_len"] != 32:
        raise ValueError("checkpoint is not the expected four-block width-64 context-32 model")
    x, b = collect_sites(module, steps, 42, horizon)
    xt, bt = collect_sites(module, steps, 43, horizon)
    maps, probes = [], {}
    for i, name in enumerate(SITES):
        mapping = fit_map(x[:, i], b)
        maps.append(mapping)
        predicted = xt[:, i] @ mapping.decoder + mapping.bias
        sse = np.sum((predicted-bt[:, :2])**2)
        sst = np.sum((bt[:, :2]-bt[:, :2].mean(0))**2)
        probes[name] = {"heldout_r2": float(1-sse/sst), "heldout_mse": float(sse/predicted.size)}
    def ev(site=3, *, seed=2042, episodes=confirm_episodes, **kwargs):
        return evaluate(module, maps[site], policy, site=site, seed=seed,
                        episodes=episodes, horizon=horizon, **kwargs)
    random = ev(seed=1042, episodes=tune_episodes, random_actions=True)
    tuning, selected, curves, conditions = {}, {}, {}, {}
    for mode in ("point", "mirror"):
        tuning[mode] = {}
        for l in LAMBDAS:
            tuning[mode][str(l)] = ev(seed=1042, episodes=tune_episodes, mode=mode, strength=l)
        selected[mode] = select_lambda(tuning[mode], random["occupancy"])
        print(f"seed {seed}: {mode} selected lambda {selected[mode]}", flush=True)
        curves[mode] = {}
        for l in LAMBDAS:
            curves[mode][str(l)] = ev(mode=mode, strength=l)
            print(f"seed {seed}: confirm {mode} lambda {l}: {curves[mode][str(l)]['occupancy']:.6f}", flush=True)
        for site, name in enumerate(SITES):
            conditions[f"{name}/{mode}"] = (curves[mode][str(selected[mode])] if site == 3 else
                ev(site, mode=mode, strength=selected[mode]))
        conditions[f"orthogonal/{mode}"] = ev(mode=mode, strength=selected[mode], matched_null=True)
    conditions["intact"] = curves["point"]["0.0"]
    conditions["random"] = ev(random_actions=True)
    report = {"training_seed": seed, "provenance": provenance,
              "experiment_revision": revision(STUDY),
              "harness_revision": revision(Path(inspect.getfile(load_module_only)).parent),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "prediction_source_sha256": hashlib.sha256(Path(inspect.getfile(BeliefPolicy)).read_bytes()).hexdigest(),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "prediction_protocol": {"n": policy.n, "gamma": policy.gamma, "iterations": policy.iterations,
                                      "seed": 2042, "episodes": confirm_episodes, "horizon": horizon,
                                      "filter": "exact token/action posterior; no rewards",
                                      "policy": "nearest-grid discounted value-iteration reference, not certified optimal"},
              "prediction_random": simulate(policy, "point", 0, episodes=confirm_episodes,
                                            horizon=horizon, seed=2042, random_actions=True),
              "protocol": {"fit_seed": 42, "test_seed": 43, "retained_steps_each": steps,
                           "probe_warmup": 8, "tune_seed": 1042, "confirm_seed": 2042,
                           "tune_episodes": tune_episodes, "confirm_episodes": confirm_episodes,
                           "horizon": horizon, "lambda_grid": LAMBDAS,
                           "selection": "smallest lambda <= random occupancy, otherwise lowest occupancy"},
              "uncertainty": "episode standard error within one frozen training seed",
              "probes": probes, "tuning_random": random, "tuning": tuning,
              "selected": selected, "curves": curves, "conditions": conditions,
              "prediction": prediction, "elapsed_seconds": time.monotonic()-started}
    write_report(output, report)
    return report


def aggregate(reports, prediction, output):
    def combine(results):
        keys = ("occupancy", "action_fractions", "action_change_rate", "action_given_state",
                "noop_given_reward_state", "installed_belief_policy_agreement")
        result = {}
        for key in keys:
            values = np.array([r[key] for r in results], dtype=float)
            result[key] = values.mean(0).tolist()
            result[key+"_se"] = (values.std(0, ddof=1)/np.sqrt(len(values))).tolist()
        return result
    report = {"training_seeds": [r["training_seed"] for r in reports],
              "selected": {str(r["training_seed"]): r["selected"] for r in reports},
              "uncertainty": "standard error across five independent training-seed means, not episode SE",
              "conditions": {key: combine([r["conditions"][key] for r in reports]) for key in reports[0]["conditions"]},
              "curves": {m: {str(l): combine([r["curves"][m][str(l)] for r in reports]) for l in LAMBDAS} for m in ("point", "mirror")},
              "prediction": prediction,
              "probes": {s: {"heldout_r2": float(np.mean([r["probes"][s]["heldout_r2"] for r in reports])),
                               "heldout_r2_se": float(np.std([r["probes"][s]["heldout_r2"] for r in reports], ddof=1)/np.sqrt(len(reports)))} for s in SITES},
              "per_seed_files": [f"belief_steering_fsq_seed{r['training_seed']}.json" for r in reports]}
    write_report(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = BeliefPolicy()
    budget = dict(steps=128, tune_episodes=2, confirm_episodes=2, horizon=32) if args.smoke else dict(steps=12000, tune_episodes=16, confirm_episodes=32, horizon=1024)
    prediction = prediction_curve(policy, episodes=budget["confirm_episodes"], horizon=budget["horizon"], seed=2042)
    destination = ARTIFACTS/"smoke" if args.smoke else RESULTS
    reports = []
    for seed in args.seeds:
        output = destination/f"belief_steering_fsq_seed{seed}.json"
        if output.exists() and not args.smoke:
            reports.append(json.loads(output.read_text()))
        else:
            reports.append(run_seed(seed, policy, prediction, output=output, **budget))
    if len(reports) == 5:
        aggregate(reports, prediction, destination/"belief_steering_fsq_5seed.json")


if __name__ == "__main__":
    main()
