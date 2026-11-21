"""Post-hoc held-out belief targets for the temperature-1.5 ladder only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import probe
from experiments.belief_quotient_history_jax_2026_11.continuation import (
    fetch_parent_checkpoint, load_params, locate_parent_run,
)
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as base
from experiments.belief_quotient_history_jax_lumped_temp15_2026_11.shared import STUDY

TARGET_DEFINITIONS = {
    "belief": "Exact three-state token/action filtered belief b",
    "reward_state": "rho = b[2] (paper state 3)",
    "state_distinction": "delta = b[0] - b[1] (paper states 1 minus 2)",
    "coarse_reward_class": "c[B]: exact quotient filter A={0,1}, B={2}; tokens {0,1}->not-C, 2->C",
}


def quotient_matrices(process):
    transitions = np.asarray(process.transitions, dtype=float)
    coarse_t = np.stack((transitions[..., :2].sum(-1), transitions[..., 2]), -1)
    emission = np.asarray(process.emission, dtype=float)
    coarse_e = np.stack((emission[:, :2].sum(-1), emission[:, 2]), -1)
    if not (np.allclose(coarse_t[:, 0], coarse_t[:, 1], rtol=0, atol=1e-7)
            and np.allclose(coarse_e[0], coarse_e[1], rtol=0, atol=1e-7)):
        raise ValueError("process is not exactly lumpable into {0,1}, {2}")
    initial = np.asarray(process.initial, dtype=float)
    return np.array([initial[:2].sum(), initial[2]]), coarse_t[:, [0, 2]], coarse_e[[0, 2]]


def coarse_filter(process, tokens, actions):
    """Episode-major decision-time distributions, including the reset row."""
    initial, transitions, emission = quotient_matrices(process)
    tokens, actions = np.asarray(tokens), np.asarray(actions)
    if tokens.ndim != 2 or tokens.shape != actions.shape or tokens.shape[1] == 0:
        raise ValueError("tokens/actions must have matching nonempty (episodes, time) shapes")
    result = np.empty((*tokens.shape, 2))
    belief = np.broadcast_to(initial, (len(tokens), 2)).copy()
    for t in range(tokens.shape[1]):
        if t:
            belief = np.einsum("bi,bij->bj", belief, transitions[actions[:, t - 1]])
        belief *= emission[:, (tokens[:, t] == 2).astype(int)].T
        belief /= belief.sum(-1, keepdims=True)
        result[:, t] = belief
    return result


def collect_targets(spec, process, params, key, *, rows, warmup, episode_length, coarse):
    episodes = -(-rows // (episode_length - warmup))
    records = probe._probe_rollout(spec, process, params, key, episodes, episode_length)

    def flatten(value):
        value = np.asarray(value)[:, warmup:]
        return value.reshape(-1, *value.shape[2:])[:rows]

    belief = flatten(records["belief"])
    targets = {"belief": belief, "reward_state": belief[:, 2:3],
               "state_distinction": belief[:, 0:1] - belief[:, 1:2]}
    if coarse:
        tokens = np.asarray(records["obs"])[..., :process.n_tokens].argmax(-1)
        targets["coarse_reward_class"] = flatten(
            coarse_filter(process, tokens, records["action"]))[:, 1:2]
    return flatten(records["embedding"]), targets, float(flatten(records["reward"]).mean())


def reprobe_checkpoint(spec, process, params, key, *, smoke=False, episode_length=1024, coarse=True):
    budget = probe.probe_budget(smoke)
    fit_key, test_key = jax.random.split(key)
    kwargs = dict(warmup=budget["warmup"], episode_length=episode_length, coarse=coarse)
    fit_x, fit_y, _ = collect_targets(spec, process, params, fit_key, rows=budget["train_rows"], **kwargs)
    test_x, test_y, reward = collect_targets(spec, process, params, test_key, rows=budget["test_rows"], **kwargs)
    metrics = {}
    for name, target in fit_y.items():
        weight, bias = probe.fit_affine_probe(fit_x, target, ridge=probe.PROBE_RIDGE)
        predicted = probe.probe_predict(weight, bias, test_x)
        metrics[name] = {**probe.global_mse_metrics(predicted, test_y[name]),
                         "r_squared": probe.r2_score(predicted, test_y[name])}
    return {"reward_per_step": reward, "targets": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf", required=True, choices=[f"variant_{v}_{a}" for v in (1, 2, 3) for a in base.ARCHITECTURES])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", nargs="+", type=int, help="Optional explicit checkpoint subset")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    leaf = Path(__file__).parent / args.leaf
    parent = locate_parent_run(leaf, study=STUDY, seed=42)
    result_dir = leaf / "results" / parent["run_id"]
    summary_path = result_dir / "condition_summary.json"
    archived = json.loads(summary_path.read_text())["checkpoint_reports"]
    if args.steps is not None and not set(args.steps) <= {r["checkpoint_step"] for r in archived}:
        parser.error("--steps must name archived checkpoints")
    out = args.out or result_dir / "reprobe_targets.json"
    if out.exists():
        raise FileExistsError(out)
    variant = int(args.leaf.split("_")[1])
    process = base.ladder_process(variant)
    spec = base.model_spec(parent["architecture"], process)
    coarse = variant in (1, 2)
    if coarse:
        quotient_matrices(process)
    key = jax.random.fold_in(jax.random.PRNGKey(42), 2)
    reports = []
    for old in archived:
        step = old["checkpoint_step"]
        if args.steps is not None and step not in args.steps:
            continue
        start = time.perf_counter()
        checkpoint_file = f"checkpoints/step_{step:010d}.npz"
        path = fetch_parent_checkpoint({**parent, "checkpoint_file": checkpoint_file}, leaf,
                                      leaf / "artifacts" / parent["run_id"] / checkpoint_file)
        report = reprobe_checkpoint(spec, process, load_params(path, spec), key, smoke=args.smoke, coarse=coarse)
        difference = abs(report["targets"]["belief"]["r_squared"] - old["targets"]["belief"]["r_squared"])
        report.update(checkpoint_step=step, archived_belief_r_squared=old["targets"]["belief"]["r_squared"],
                      belief_r_squared_abs_difference=difference, seconds=time.perf_counter() - start,
                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reports.append(report)
        print(json.dumps(report), flush=True)
    output = {"study": STUDY, "leaf": args.leaf, "run_id": parent["run_id"], "seed": 42,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_run": parent, "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              "target_definitions": TARGET_DEFINITIONS, "coarse_omitted_variants": [3],
              "coarse_equivalence": "Uniform nonreward destinations and symmetric emissions make A/B tokens carry identical reward-class likelihood; c[B] equals rho for variants 1 and 2.",
              "smoke": args.smoke, "budget": probe.probe_budget(args.smoke), "representation": "embedding",
              "probe_key": "fold_in(PRNGKey(42), 2)", "probe_ridge": probe.PROBE_RIDGE,
              "checkpoint_reports": reports,
              "max_abs_belief_r_squared_difference": max(r["belief_r_squared_abs_difference"] for r in reports)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"max_abs_belief_r_squared_difference={output['max_abs_belief_r_squared_difference']:.12g}", flush=True)


if __name__ == "__main__":
    main()
