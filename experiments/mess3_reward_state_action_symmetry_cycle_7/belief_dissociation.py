"""Offline rho/delta dissociation campaign; outputs never replace FSQ reports."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from experiments.mess3_reward_state_action_symmetry_cycle_7 import belief_steering_followup as steering
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_prediction import (
    BeliefPolicy, LAMBDAS, prediction_curve,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import WorkItem, _download_module

MODES = ("point", "mirror", "rho_flip", "rho_point")
LABELS = {2: "rsq", 3: "fsq"}
STUDY = steering.STUDY
ARTIFACTS = STUDY/"artifacts/belief_dissociation"
RESULTS = STUDY/"results"


def write(output, report, markdown):
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    output.with_suffix(".md").write_text(markdown+"\n")


def restore(variant, seed):
    condition = f"variant_{variant}_ctx32_ent"
    run_id = f"mess3_reward_state_action_symmetry_cycle_6-{condition}-seed{seed}-10m"
    run_dir = STUDY/condition/"results"/run_id
    curve = run_dir/"checkpoint_probe_curve.json"
    record = json.loads(curve.read_text())["checkpoints"][-1]
    module_path = ARTIFACTS/run_id/"default_policy"
    item = WorkItem(condition, variant, seed, "final", 6, record.get("training_iteration"),
                    int(record["agent_steps"]), run_id, record["checkpoint_name"], curve,
                    run_dir/"run_manifest.json", module_path, ARTIFACTS/"unused.json")
    download = _download_module(item)
    if variant == 2 and seed == 42:
        expected = "6ff2b60203e298c7967cd16e2a0ad5676fbc22c288e02e2515a38a06a635cc5b"
        if download["module_files"]["module_state.pkl"]["sha256"] != expected:
            raise ValueError("RSQ seed-42 checkpoint is not the manuscript module")
    return module_path, {"run_id": run_id, "agent_steps": item.agent_steps,
                         "training_manifest": json.loads(item.run_manifest_path.read_text()), **download}


def coordinates(b):
    return np.column_stack((b[:, 0], b[:, 1], b[:, 2], b[:, 0]-b[:, 1]))


def maps_and_probes(module, variant, steps, horizon):
    x, b = steering.collect_sites(module, steps, 42, horizon, variant)
    xt, bt = steering.collect_sites(module, steps, 43, horizon, variant)
    maps, probes = [], {}
    truth = coordinates(bt)
    for i, site in enumerate(steering.SITES):
        mapping = steering.fit_map(x[:, i], b)
        maps.append(mapping)
        decoded = xt[:, i] @ mapping.decoder + mapping.bias
        predicted = coordinates(np.column_stack((decoded, 1-decoded.sum(1))))
        sse = ((predicted-truth)**2).sum(0)
        sst = ((truth-truth.mean(0))**2).sum(0)
        probes[site] = {name: {"r2": float(1-e/v) if v else None,
                              "mse": float(e/len(truth))}
                        for name, e, v in zip(("b1", "b2", "rho", "delta"), sse, sst)}
    return maps, probes


def provenance(download, budget):
    from experiments.mess3_reward_state_action_symmetry_cycle_7 import belief_pulse
    from experiments.mess3_reward_state_action_symmetry_cycle_7 import belief_steering_prediction
    paths = [Path(__file__), Path(inspect.getfile(steering)),
             Path(inspect.getfile(belief_steering_prediction)), Path(inspect.getfile(belief_pulse))]
    return {"checkpoint": download, "experiment_revision": steering.revision(STUDY),
            "harness_revision": steering.revision(Path(inspect.getfile(load_module_only)).parent),
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
            "protocol": {**budget, "fit_seed": 42, "test_seed": 43, "warmup": 8,
                         "tune_seed": 1042, "confirm_seed": 2042,
                         "reference": "N=60, gamma=.97, 800 iterations; not certified optimal"}}


def run_dissociation(module, maps, probes, policy, prediction, budget):
    def ev(site=3, *, seed=2042, episodes=budget["confirm_episodes"], **kwargs):
        return steering.evaluate(module, maps[site], policy, site=site, seed=seed,
                                 episodes=episodes, horizon=budget["horizon"], variant=policy.variant, **kwargs)
    random = ev(seed=1042, episodes=budget["tune_episodes"], random_actions=True)
    tuning, selected, curves, conditions = {}, {}, {}, {}
    for mode in MODES:
        tuning[mode] = {str(l): ev(seed=1042, episodes=budget["tune_episodes"], mode=mode, strength=l) for l in LAMBDAS}
        selected[mode] = steering.select_lambda(tuning[mode], random["occupancy"])
        curves[mode] = {str(l): ev(mode=mode, strength=l) for l in LAMBDAS}
        for site, name in enumerate(steering.SITES):
            conditions[f"{name}/{mode}"] = (curves[mode][str(selected[mode])] if site == 3 else
                ev(site, mode=mode, strength=selected[mode]))
        conditions[f"orthogonal/{mode}"] = ev(mode=mode, strength=selected[mode], matched_null=True)
        print(f"{LABELS[policy.variant]} {mode}: selected {selected[mode]}, lambda=1 occupancy {curves[mode]['1.0']['occupancy']:.6f}", flush=True)
    conditions["intact"] = curves["point"]["0.0"]
    conditions["random"] = ev(random_actions=True)
    return {"probes": probes, "tuning_random": random, "tuning": tuning, "selected": selected,
            "curves": curves, "conditions": conditions, "prediction": prediction}


def mean_se(values):
    a = np.asarray(values, dtype=float)
    return {"mean": a.mean(0).tolist(), "se": (a.std(0, ddof=1)/np.sqrt(len(a))).tolist()}


METRICS = ("occupancy", "noop_given_reward_state", "action_change_rate", "action_fractions",
           "action_given_state", "installed_belief_policy_agreement", "mean_shift_norm",
           "mean_activation_norm", "shift_relative_to_mean_activation")


def summarize(reports):
    return {"selected": {str(r["training_seed"]): r["selected"] for r in reports},
            "curves": {m: {str(l): {k: mean_se([r["curves"][m][str(l)][k] for r in reports]) for k in METRICS}
                            for l in LAMBDAS} for m in MODES},
            "conditions": {c: {k: mean_se([r["conditions"][c][k] for r in reports]) for k in METRICS}
                           for c in reports[0]["conditions"]},
            "probes": {s: {c: mean_se([r["probes"][s][c]["r2"] for r in reports])
                           for c in ("b1", "b2", "rho", "delta")} for s in steering.SITES},
            "prediction": reports[0]["prediction"]}


def report_md(report):
    lines = ["# Rho/delta steering dissociation", "",
             "Occupancy includes all decisions. Targets use raw affine decoding; no clipping.",
             "delta = b1-b2; rho = b3. Action order: Noop, Tilt X, Tilt Y.",
             "Reference: near-optimal grid policy, not certified optimal. Aggregate SE is across training seeds.",
             "Selected lambdas: "+json.dumps(report.get("selected", {})), "",
             "| Intervention | lambda | Occupancy | Noop given state 3 | Action change | Mean shift norm | Shift / mean activation | Predicted occupancy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, curve in report["curves"].items():
        for l, result in curve.items():
            def value(key):
                v = result[key]
                return f"{v['mean']:.6g} ± {v['se']:.3g}" if isinstance(v, dict) else f"{v:.6g}"
            lines.append(f"| {mode} | {l} | "+" | ".join(value(k) for k in ("occupancy", "noop_given_reward_state", "action_change_rate", "mean_shift_norm", "shift_relative_to_mean_activation"))+f" | {report['prediction'][mode][l]['occupancy']:.6g} |")
    lines += ["", "Full selected-lambda site, orthogonal-control and coordinate-R² tables are in the JSON."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    args = parser.parse_args()
    torch.set_num_threads(1)
    budget = (dict(steps=128, tune_episodes=2, confirm_episodes=2, horizon=64) if args.smoke else
              dict(steps=12000, tune_episodes=16, confirm_episodes=32, horizon=1024))
    from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_pulse import run_pulses, aggregate_pulses, pulse_md
    all_reports = {}
    for variant in (2, 3):
        label = LABELS[variant]
        policy = BeliefPolicy(variant=variant)
        predictions = prediction_curve(policy, modes=MODES, episodes=budget["confirm_episodes"], horizon=budget["horizon"], seed=2042)
        reports, pulse_reports = [], []
        for seed in args.seeds:
            start = time.monotonic()
            path, download = restore(variant, seed)
            destination = ARTIFACTS/"smoke" if args.smoke else STUDY/f"variant_{variant}_ctx32_ent/results"
            name = f"belief_steering_{label}{'_rho' if variant == 3 else ''}_seed{seed}"
            output = destination/(name+".json")
            pulse_output = destination/f"belief_pulse_{label}_seed{seed}.json"
            if output.exists() or pulse_output.exists():
                raise FileExistsError("Choose new output names or move smoke outputs; reports are never overwritten")
            module = load_module_only(path).to("cpu").eval()
            if len(module.encoder.blocks) != 4 or module.model_config["context_len"] != 32 or module.model_config["d_model"] != 64:
                raise ValueError("unexpected model architecture")
            maps, probes = maps_and_probes(module, variant, budget["steps"], budget["horizon"])
            prov = provenance(download, budget)
            report = {"agent": label, "training_seed": seed, "provenance": prov,
                      **run_dissociation(module, maps, probes, policy, predictions, budget)}
            report["elapsed_seconds"] = time.monotonic()-start
            write(output, report, report_md(report))
            reports.append(report)
            pulse = {"agent": label, "training_seed": seed, "provenance": prov,
                     **run_pulses(module, maps, variant, smoke=args.smoke)}
            write(pulse_output, pulse, pulse_md(pulse))
            pulse_reports.append(pulse)
            print(f"{label} seed {seed} finished in {time.monotonic()-start:.1f}s", flush=True)
        all_reports[label] = summarize(reports) if len(reports) > 1 else reports[0]
        if len(reports) == 5:
            destination = ARTIFACTS/"smoke" if args.smoke else RESULTS
            pulse_aggregate = aggregate_pulses(pulse_reports)
            write(destination/f"belief_pulse_{label}_5seed.json", pulse_aggregate, pulse_md(pulse_aggregate))
    if len(args.seeds) == 5:
        aggregate = {"agents": all_reports, "uncertainty": "SE across five training seeds",
                     "per_seed_reports": [f"variant_{v}_ctx32_ent/results/belief_steering_{LABELS[v]}{'_rho' if v == 3 else ''}_seed{s}.json" for v in (2, 3) for s in args.seeds]}
        destination = ARTIFACTS/"smoke" if args.smoke else RESULTS
        write(destination/"belief_steering_dissociation_5seed.json", aggregate,
              "\n\n".join(f"## {label.upper()}\n"+report_md(report) for label, report in all_reports.items()))


if __name__ == "__main__":
    main()
