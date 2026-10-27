"""Verified PPO encoder warm starts for pure-JAX guess-or-pass."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from harness.cli import execute_experiment, load_experiment, make_run_context
from ray.rllib.core.rl_module.rl_module import RLModule

from experiments.pusher_b10_guess_or_pass_2026_10.transfer import (
    SOURCE_CATALOG,
    _local_source_module,
    fetch_source_module,
)
from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.process import BOS_TOKEN

from .convert import torch_encoder_to_jax
from .ppo import OVERRIDES_ENV
from .process import make_env

PPO_SOURCES = tuple(
    source
    for source, metadata in SOURCE_CATALOG.items()
    if metadata["algorithm"] == "ppo"
)
STUDY_MODULE = "experiments.pusher_b10_guess_or_pass_jax_2026_10.transfer"


def _parity_check(module, trunk: dict, *, cost: float, seed: int) -> float:
    env = make_env(cost)
    sample_key = jax.random.key(seed ^ 0x47504A58)
    token_sequences = env.sample_tokens(
        sample_key,
        batch_size=16,
        length=env.episode_length,
    )
    model_tokens = jnp.concatenate(
        [
            jnp.full((16, 1), BOS_TOKEN, dtype=jnp.int32),
            token_sequences[:, :-1],
        ],
        axis=1,
    )
    observations = jnp.concatenate(
        [
            jnp.zeros((16, 1, 2), dtype=jnp.float32),
            jax.nn.one_hot(token_sequences[:, :-1], 2, dtype=jnp.float32),
        ],
        axis=1,
    )
    with torch.no_grad():
        torch_observations = torch.tensor(
            np.asarray(observations), dtype=torch.float32, device="cpu"
        )
        source_activations = module.encoder.forward_complete_episode(
            torch_observations
        )
    with jax.default_matmul_precision("highest"):
        jax_activations = tm.encode(tm.ModelSpec(), trunk, model_tokens)
    difference = np.max(
        np.abs(
            source_activations.detach().cpu().numpy()
            - np.asarray(jax_activations)
        )
    )
    if not np.isfinite(difference) or difference >= 1e-4:
        raise RuntimeError(
            f"source/JAX encoder parity failed: max abs diff={difference:.8g}"
        )
    return float(difference)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_module")
    parser.add_argument("--source", choices=PPO_SOURCES, required=True)
    parser.add_argument("--trunk", choices=("full", "frozen"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hardware-profile", "--hardware", default="auto")
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)

    if args.source.startswith("sac_"):
        parser.error("only PPO transfer sources are supported")
    if "." not in args.experiment_module:
        args.experiment_module = (
            "experiments.pusher_b10_guess_or_pass_jax_2026_10."
            f"{args.experiment_module}.experiment"
        )
    parts = args.experiment_module.split(".")
    leaf = parts[-2] if parts[-1] == "experiment" else parts[-1]
    if leaf not in {"ppo_c2", "ppo_c1p5", "ppo_c1"}:
        parser.error(f"unsupported JAX guess-or-pass leaf: {leaf}")

    source_spec: dict[str, Any] = SOURCE_CATALOG[args.source]
    experiment = load_experiment(args.experiment_module)
    if args.source_dir is None:
        module_dir, checkpoint_path = fetch_source_module(
            source_label=args.source,
            destination=(
                Path(__file__).resolve().parent
                / "artifacts"
                / "sources"
                / args.source
            ),
        )
    else:
        module_dir = _local_source_module(args.source_dir)
        checkpoint_path = str(args.source_dir.expanduser().resolve())
    source_module = RLModule.from_checkpoint(str(module_dir))
    if hasattr(source_module, "to"):
        source_module.to("cpu")
    source_state = source_module.get_state()
    spec = tm.ModelSpec()
    trunk = torch_encoder_to_jax(source_state, spec)
    encoder_keys = sorted(key for key in source_state if key.startswith("encoder."))
    ignored_source_keys = sorted(
        key for key in source_state if not key.startswith("encoder.")
    )
    parity = _parity_check(
        source_module,
        trunk,
        cost={"ppo_c2": 2.0, "ppo_c1p5": 1.5, "ppo_c1": 1.0}[leaf],
        seed=args.seed,
    )
    metadata: dict[str, Any] = {
        "source": args.source,
        "run_id": source_spec["run_id"],
        "checkpoint_path": checkpoint_path,
        "source_module_directory": str(module_dir),
        "source_env_steps": source_spec["env_steps"],
        "source_probe_1_minus_r2": source_spec["probe_1_minus_r2"],
        "pretraining_objective": source_spec["pretraining_objective"],
        "trunk_mode": args.trunk,
        "encoder_parity_max_abs_diff": parity,
        "encoder_keys": encoder_keys,
        "ignored_source_keys": ignored_source_keys,
    }
    os.environ[OVERRIDES_ENV] = json.dumps(metadata)
    context = make_run_context(
        experiment,
        seed=args.seed,
        run_id=args.run_id,
        smoke=args.smoke,
        resume_from=module_dir,
        hardware_profile=args.hardware_profile,
    )
    command = [
        sys.executable,
        "-m",
        STUDY_MODULE,
        args.experiment_module,
        "--source",
        args.source,
        "--trunk",
        args.trunk,
        "--seed",
        str(args.seed),
        "--hardware-profile",
        args.hardware_profile,
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.source_dir:
        command.extend(["--source-dir", str(args.source_dir)])
    if args.smoke:
        command.append("--smoke")
    if args.upload_artifacts is not None:
        command.append(
            "--upload-artifacts"
            if args.upload_artifacts
            else "--no-upload-artifacts"
        )
    execute_experiment(
        experiment,
        context,
        command=command,
        runtime_overrides={
            **metadata,
            "run_id": context.run_id,
            "source_dir": str(args.source_dir) if args.source_dir else None,
            "resume_from": str(module_dir),
            OVERRIDES_ENV: os.environ[OVERRIDES_ENV],
        },
        upload_artifacts=args.upload_artifacts,
    )


if __name__ == "__main__":
    main()
