"""Short throughput/learning runs of the JAX trainers; one JSON line per record.

    python -m experiments.pusher_b_jax_2026_10.benchmark ce --steps 3000 --out ce.jsonl
    python -m experiments.pusher_b_jax_2026_10.benchmark ppo --env-steps 3000000 --out ppo.jsonl

``--precision highest`` disables TF32 matmuls (PyTorch's fp32 default).
The legacy counterparts are ``legacy_short/{ce_b10,ppo_b10}`` (run via
``harness.cli`` with ``experiments/pusher_b`` checked out).
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ce", "ppo"])
    parser.add_argument("--preset", default="b10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--env-steps", type=int, default=3_000_000)
    parser.add_argument("--precision", choices=["default", "high", "highest"], default="default")
    parser.add_argument("--num-envs", type=int, default=2_048)
    parser.add_argument("--minibatch-episodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=None, help="constant lr override")
    parser.add_argument("--rollout-chunks", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import jax

    jax.config.update("jax_default_matmul_precision", args.precision)
    with open(args.out, "w") as handle:

        def log(record):
            handle.write(json.dumps({"kind": "record", **record}) + "\n")
            handle.flush()

        if args.mode == "ce":
            from experiments.pusher_b_jax_2026_10 import supervised

            config = supervised.SupervisedConfig(
                total_steps=args.steps,
                eval_steps=tuple(s for s in supervised.CHECKPOINT_STEPS if s < args.steps)
                + (args.steps,),
            )
            result = supervised.train(args.preset, seed=args.seed, config=config, log=log)
            result.pop("params")
            result.pop("history")
        else:
            from experiments.pusher_b_jax_2026_10 import ppo

            overrides = {}
            if args.lr is not None:
                overrides["lr_schedule"] = ((0, args.lr),)
            config = ppo.PPOConfig(
                total_env_steps=args.env_steps,
                num_envs=args.num_envs,
                minibatch_episodes=args.minibatch_episodes,
                num_epochs=args.epochs,
                rollout_chunks=args.rollout_chunks,
                **overrides,
            )
            _, _, result = ppo.train(args.preset, seed=args.seed, config=config, log=log)
        summary = {"kind": "summary", **vars(args), **result}
        handle.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
