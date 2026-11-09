"""Throughput and batch-size benchmark for the JAX history studies.

Uses each study's real recipe (entropy, minibatches) and overrides only
``num_envs`` and the number of iterations. Examples::

    python -m experiments.belief_quotient_history_jax_2026_11.benchmark \
        --study both --iterations 4 --out bench.json
    python -m experiments.belief_quotient_history_jax_2026_11.benchmark \
        --study both --max-batch --out max_batch.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import subprocess
import sys
import time

import jax

from experiments.belief_quotient_history_jax_2026_11 import shared as ladder
from experiments.belief_quotient_history_jax_2026_11.reinforce import train
from experiments.reward_factors_history_jax_2026_11 import process as factors
from experiments.reward_factors_history_jax_2026_11 import shared as factor_shared

STUDIES = {
    "ladder": ("variant_3", lambda: ladder.ladder_process(3), ladder.model_spec,
               ladder.train_config),
    "factors": ("reward_both", lambda: factors.factor_process("reward_both"),
                factor_shared.model_spec, factor_shared.train_config),
}


def _studies(study):
    return ("ladder", "factors") if study == "both" else (study,)


def run_arm(study, architecture, num_envs, iterations, seed):
    condition, make_process, make_spec, make_config = STUDIES[study]
    process = make_process()
    spec = make_spec(architecture, process)
    recipe = make_config(False)
    config = replace(recipe, num_envs=num_envs,
                     total_env_steps=iterations * num_envs * recipe.episode_length)
    start = time.perf_counter()
    _, curves, summary = train(spec, process, config, seed=seed)
    steady = summary["steady_env_steps_per_second"] or summary["env_steps_per_second"]
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "study": study, "condition": condition, "architecture": architecture,
        "num_envs": num_envs, "num_minibatches": config.num_minibatches,
        "minibatch_steps": config.batch_steps // config.num_minibatches,
        "wall_seconds": time.perf_counter() - start,
        "full_budget_env_steps": recipe.total_env_steps,
        "full_budget_iterations": -(-recipe.total_env_steps // config.batch_steps),
        "projected_full_budget_seconds": recipe.total_env_steps / steady + summary["compile_seconds"],
        "final_reward_per_step": curves[-1]["reward_per_step"],
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        **summary,
    }


def _physical_bytes():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0]) * 2**20


def _probe(study, architecture, num_envs, seed):
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}
    cmd = [sys.executable, "-m", __spec__.name, "--probe-one", study, architecture,
           str(num_envs), "--seed", str(seed)]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("{"):
            return {"ok": True, **json.loads(line)}
    oom = "RESOURCE_EXHAUSTED" in done.stderr or "Out of memory" in done.stderr
    return {"ok": False, "num_envs": num_envs, "oom": oom, "stderr_tail": done.stderr[-400:]}


def max_batch(study, architecture, start, cap, margin, seed):
    """Double ``num_envs`` until OOM or ``cap``; recommend the largest multiple of
    32 whose (linearly interpolated) peak stays within ``margin`` of VRAM."""

    total = _physical_bytes()
    tries, num_envs = [], start
    while num_envs <= cap:
        result = _probe(study, architecture, num_envs, seed)
        tries.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "stderr_tail"}), flush=True)
        if not result["ok"]:
            break
        num_envs *= 2
    ok = [t for t in tries if t["ok"]]
    budget = margin * total
    recommended = max([t["num_envs"] for t in ok if t["peak_bytes_in_use"] <= budget], default=None)
    if len(ok) >= 2 and ok[-1]["peak_bytes_in_use"] < budget and not tries[-1]["ok"]:
        (n1, p1), (n2, p2) = [(t["num_envs"], t["peak_bytes_in_use"]) for t in ok[-2:]]
        estimate = int(n2 + (budget - p2) * (n2 - n1) / max(1, p2 - p1)) // 32 * 32
        recommended = max(recommended, min(estimate, tries[-1]["num_envs"] - 32))
    return {"study": study, "architecture": architecture, "physical_bytes": total,
            "margin": margin, "largest_ok": max([t["num_envs"] for t in ok], default=None),
            "first_failure": next((t["num_envs"] for t in tries if not t["ok"]), None),
            "recommended_num_envs": recommended, "tries": tries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["ladder", "factors", "both"], default="both")
    parser.add_argument("--architectures", nargs="+", default=list(ladder.ARCHITECTURES))
    parser.add_argument("--num-envs", type=int, default=352)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batch", action="store_true")
    parser.add_argument("--cap", type=int, default=16_384)
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--probe-one", nargs=3, metavar=("STUDY", "ARCH", "NUM_ENVS"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.probe_one:
        study, architecture, num_envs = args.probe_one
        print(json.dumps(run_arm(study, architecture, int(num_envs), 2, args.seed)), flush=True)
        return
    rows = []
    for study in _studies(args.study):
        for architecture in args.architectures:
            if args.max_batch:
                row = max_batch(study, architecture, args.num_envs, args.cap, args.margin, args.seed)
            else:
                row = run_arm(study, architecture, args.num_envs, args.iterations, args.seed)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "tries"}), flush=True)
    result = {"devices": [str(d) for d in jax.devices()], "jax": jax.__version__, "rows": rows}
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
