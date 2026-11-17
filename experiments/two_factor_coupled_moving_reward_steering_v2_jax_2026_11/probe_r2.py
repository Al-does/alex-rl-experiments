"""Held-out decoder R² for the steering maps of one source study.

Refits the decoders exactly as ``steering.analyze`` does (seed 42, 60,000
fitting rows) and scores them on 80,000 independent rows (seed 43), the
convention of the PR #226 reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11 import steering
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11 import steering as base


def probe_r2(run: dict, checkpoint: Path, seed: int) -> dict:
    recipe = json.loads((run["results"] / "resolved_recipe.json").read_text())
    spec = models.ModelSpec(**recipe["model"])
    process = steering.source_process(run["study"], run["condition"])
    params = base.load_params(checkpoint, spec)
    fit = base.collect_rows(spec, params, process, rows=base.FIT_ROWS, seed=seed)
    test = base.collect_rows(spec, params, process, rows=base.TEST_ROWS, seed=seed + 1)
    fit_targets, test_targets = base.map_targets(fit["belief"]), base.map_targets(test["belief"])
    names = [f"block_{i + 1}" for i in range(spec.n_layers)] + ["post_final_norm"]
    out = {}
    for site, name in enumerate(names):
        features = fit["sites"][:, site].astype(np.float64)
        out[name] = {}
        for target in steering.TARGETS:
            decoder, bias = base.affine_fit(features, fit_targets[target].astype(np.float64))
            predicted = test["sites"][:, site] @ decoder + bias
            out[name][target] = base.r_squared(predicted, test_targets[target])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=sorted(steering.SOURCE_STUDIES))
    parser.add_argument("--checkpoints", type=Path, default=base.STUDY_DIR / "artifacts" / "checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="substring filter on 'seed<seed>'")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"study": args.study, "fit_seed": args.seed, "test_seed": args.seed + 1,
              "fit_rows": base.FIT_ROWS, "test_rows": base.TEST_ROWS, "agents": {}}
    for run in steering.source_runs():
        if run["study"] != args.study or run["condition"] != "reward_both":
            continue
        if args.only and args.only not in f"seed{run['seed']}":
            continue
        report["agents"][f"seed{run['seed']}"] = probe_r2(
            run, steering.checkpoint_path(run, args.checkpoints), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
