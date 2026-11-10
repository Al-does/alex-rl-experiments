"""Shared run loop: JAX training, parameter checkpoints, then probes."""

from __future__ import annotations

from dataclasses import asdict
import json

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.probe import probe_checkpoint
from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig, train
from harness.artifacts import RunArtifacts

STEP_CHECKPOINT_INTERVAL = 2_000_000
SMOKE_NUM_ENVS = 2  # two 2,048-step iterations = the RLlib smoke target of 4,096


def checkpoint_steps(config: TrainConfig, step_offset: int = 0) -> list[int]:
    """Absolute checkpoint steps: log-spaced iterations and every 2M steps,
    both counted from ``step_offset`` (the parent's final step when continuing)."""

    log_spaced = [config.batch_steps * 2**k for k in range(config.iterations.bit_length())]
    interval = range(STEP_CHECKPOINT_INTERVAL, config.total_env_steps, STEP_CHECKPOINT_INTERVAL)
    return sorted(step_offset + s for s in set(log_spaced) | set(interval))


def _save(path, params):
    np.savez(path, **{
        jax.tree_util.keystr(key): np.asarray(value)
        for key, value in jax.tree_util.tree_leaves_with_path(params)
    })


def run_study(context, *, study, condition, architecture, process, spec, config,
              targets, representation, recipe_extra, continue_from=None):
    """``continue_from`` (see ``continuation.load_parent_checkpoint``) restarts
    training from a finished run's final parameters; ``config.total_env_steps``
    is then the *additional* budget and all recorded steps are absolute."""

    if context.seed is None:
        raise ValueError("these studies require a resolved runtime seed")
    if context.resume_from is not None:
        raise ValueError("JAX history studies always train from scratch")
    if context.smoke:
        config = TrainConfig(**{**asdict(config), "num_envs": SMOKE_NUM_ENVS,
                                "num_minibatches": min(config.num_minibatches, SMOKE_NUM_ENVS)})
    step_offset = int(continue_from["step_offset"]) if continue_from else 0
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", {
        "study": study,
        "condition": condition,
        "architecture": architecture,
        "seed": context.seed,
        "smoke": context.smoke,
        "framework": "jax",
        "jax_version": jax.__version__,
        "model": asdict(spec),
        "training": {**asdict(config), "step_offset": step_offset,
                     "total_env_steps_absolute": step_offset + config.total_env_steps},
        "continued_from": continue_from["parent"] if continue_from else None,
        "process": {name: np.asarray(value).tolist() for name, value in process._asdict().items()},
        **recipe_extra,
    })
    checkpoint_dir = context.artifacts_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    def on_checkpoint(steps, params):
        saved[steps] = jax.device_get(params)
        _save(checkpoint_dir / f"step_{steps:010d}.npz", saved[steps])

    params, curves, training = train(
        spec, process, config, seed=context.seed,
        checkpoint_steps=checkpoint_steps(config, step_offset), on_checkpoint=on_checkpoint,
        log=outputs.append_result,
        initial_params=continue_from["params"] if continue_from else None,
        step_offset=step_offset,
    )
    with open(context.results_dir / "training_curves.jsonl", "w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in curves)
    probe_root = jax.random.fold_in(jax.random.PRNGKey(context.seed), 2)
    reports = []
    for steps, checkpoint in sorted(saved.items()):
        # Same probe streams at every checkpoint: changes reflect parameters only.
        report = probe_checkpoint(
            spec, process, checkpoint, probe_root, smoke=context.smoke,
            episode_length=config.episode_length, representation=representation,
            targets=targets,
        )
        reports.append({"checkpoint_step": steps, "is_untrained": steps == 0,
                        "is_restored": bool(continue_from) and steps == step_offset, **report})
    summary = {
        "study": study, "condition": condition, "architecture": architecture,
        "seed": context.seed, "smoke": context.smoke, "training": training,
        "continued_from": continue_from["parent"] if continue_from else None,
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
