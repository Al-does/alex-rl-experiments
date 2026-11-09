"""Throughput benchmark for the JAX history studies (no probes/checkpoints).

Example::

    python -m experiments.belief_quotient_history_jax_2026_11.benchmark \
        --study both --iterations 4 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import time

import jax

from experiments.belief_quotient_history_jax_2026_11 import shared as ladder
from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig, train
from experiments.reward_factors_history_jax_2026_11 import process as factors
from experiments.reward_factors_history_jax_2026_11 import shared as factor_shared


def _arms(study):
    if study in ("ladder", "both"):
        yield "ladder", "variant_3", ladder.ladder_process(3), ladder.model_spec, ladder.TOTAL_ENV_STEPS
    if study in ("factors", "both"):
        yield ("factors", "reward_both", factors.factor_process("reward_both"),
               factor_shared.model_spec, factor_shared.TOTAL_ENV_STEPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["ladder", "factors", "both"], default="both")
    parser.add_argument("--architectures", nargs="+", default=list(ladder.ARCHITECTURES))
    parser.add_argument("--num-envs", type=int, default=352)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows = []
    for study, condition, process, make_spec, full_steps in _arms(args.study):
        for architecture in args.architectures:
            spec = make_spec(architecture, process)
            config = TrainConfig(
                total_env_steps=args.iterations * args.num_envs * ladder.EPISODE_LENGTH,
                num_envs=args.num_envs, episode_length=ladder.EPISODE_LENGTH,
                entropy_schedule=((0, 0.03),),
            )
            start = time.perf_counter()
            _, curves, summary = train(spec, process, config, seed=args.seed)
            steady = summary["steady_env_steps_per_second"] or summary["env_steps_per_second"]
            row = {
                "study": study, "condition": condition, "architecture": architecture,
                "num_envs": args.num_envs, "wall_seconds": time.perf_counter() - start,
                "full_budget_env_steps": full_steps,
                "projected_full_budget_seconds": full_steps / steady + summary["compile_seconds"],
                "final_reward_per_step": curves[-1]["reward_per_step"],
                **summary,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {"devices": [str(d) for d in jax.devices()], "jax": jax.__version__, "rows": rows}
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
