"""Time two research-size PPO rollouts without publishing pilot results."""
import argparse
from dataclasses import replace
import json

import jax

from .shared import recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("separate", "shared_wide"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("full-batch preflight requires a JAX GPU backend")
    process, config, spec, trainer = recipe(3, args.arm, False)
    config = replace(config, total_env_steps=2 * config.batch_steps)
    print(f"PPO full-batch preflight: seed={args.seed} arm={args.arm}", flush=True)
    _, _, summary = trainer(spec, process, config, seed=args.seed,
                           log=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"preflight": args.arm, "training": summary,
                      "device_memory": jax.devices()[0].memory_stats()}), flush=True)


if __name__ == "__main__":
    main()
