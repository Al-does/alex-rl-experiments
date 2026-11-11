"""Run loop for the PPO studies: JAX training, parameter checkpoints, probes.

Same artifact layout, checkpoint schedule and probe battery as
``belief_quotient_history_jax_2026_11.training``; only the trainer differs.
Probes run the actor only (the value head is ignored by ``models.step``).
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.probe import probe_checkpoint
from experiments.belief_quotient_history_jax_2026_11.training import (
    SMOKE_NUM_ENVS, _save, checkpoint_steps,
)
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11.ppo import train
from harness.artifacts import RunArtifacts


def run_study(context, *, study, condition, architecture, process, spec, config,
              targets, representation, recipe_extra):
    if context.seed is None:
        raise ValueError("these studies require a resolved runtime seed")
    if context.resume_from is not None:
        raise ValueError("JAX history studies always train from scratch")
    if context.smoke:
        config = replace(config, num_envs=SMOKE_NUM_ENVS,
                         num_minibatches=min(config.num_minibatches, SMOKE_NUM_ENVS))
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", {
        "study": study,
        "condition": condition,
        "architecture": architecture,
        "seed": context.seed,
        "smoke": context.smoke,
        "framework": "jax",
        "algorithm": "ppo",
        "jax_version": jax.__version__,
        "model": asdict(spec),
        "value_head": "linear on post_final_layer_norm embedding (shared torso)",
        "training": asdict(config),
        "process": {name: np.asarray(value).tolist() for name, value in process._asdict().items()},
        **recipe_extra,
    })
    checkpoint_dir = context.artifacts_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    def on_checkpoint(steps, params):
        saved[steps] = jax.device_get(params)
        _save(checkpoint_dir / f"step_{steps:010d}.npz", saved[steps])

    _, curves, training = train(
        spec, process, config, seed=context.seed, checkpoint_steps=checkpoint_steps(config),
        on_checkpoint=on_checkpoint, log=outputs.append_result,
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
        reports.append({"checkpoint_step": steps, "is_untrained": steps == 0, **report})
    summary = {
        "study": study, "condition": condition, "architecture": architecture,
        "seed": context.seed, "smoke": context.smoke, "algorithm": "ppo",
        "training": training, "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
