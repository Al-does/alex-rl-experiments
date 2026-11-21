"""Run the frozen RockSample checkpoint and twin-steering protocols."""

import pickle

import jax
import numpy as np
from harness.artifacts import RunArtifacts

from experiments.rocksample_jax_2026_10 import belief_probe, checkpoint_beliefs, env, ppo
from experiments.rocksample_jax_2026_10.baseline import K, N
from experiments.rocksample_jax_2026_10.ppo_bmax_r7_cont330m import experiment as cont

from . import twin_steering as twin
from . import twin_steering_protocol as protocol


def confirm(context, spec, environment, params, raw, update):
    fit_episodes, episodes = (128, 16) if context.smoke else (512, 1024)
    fitted = twin.fit(spec, environment, params, fit_episodes, protocol.FIT_SEED)
    options = protocol.variants()
    report = {
        "update": update, "fit_seed": protocol.FIT_SEED, "eval_seed": protocol.EVAL_SEED,
        "fit_episodes": fit_episodes, "episodes": episodes,
        "site": protocol.SITE_NAMES[protocol.SITE], "rank": protocol.RANK,
        "strength": protocol.STRENGTH, "fit": protocol.jsonable(fitted),
        "cells": {}, "unavailable": {},
    }
    probe_coverage = fitted["diagnostics"]
    for rock in range(environment.k):
        for good, direction in protocol.DIRECTIONS:
            for name, config in options.items():
                key = f"{rock}|{direction}|{name}"
                kwargs = dict(config)
                random_policy = kwargs.pop("random_policy", False)
                needs_twins = kwargs.get("reflect") or kwargs.get("patch") == "pca_all"
                if needs_twins and probe_coverage[rock]["relevant"] == 0:
                    report["unavailable"][key] = "No fit decisions after a check"
                    continue
                try:
                    intervention = twin.intervention(fitted, rock, good, **kwargs)
                except np.linalg.LinAlgError:
                    if name != "legacy_clamp":
                        raise
                    report["unavailable"][key] = "Singular nuisance-controlled clamp calibration"
                    continue
                values = twin.rollout(spec, environment, params, intervention,
                                      episodes=episodes, seed=protocol.EVAL_SEED,
                                      rock=rock, good=good, random_policy=random_policy)
                report["cells"][key] = twin.summarize(values, rock)
        print(f"RockSample twin steering update {update}, rock {rock} complete", flush=True)
    path = raw.write_json(f"steering_update{update:03d}.json", report)
    cells = {
        key: {name: value for name, value in cell.items()
              if name in ("episodes", "target", "target_ci95", "other_good_counts",
                          "other_bad_counts", "mean_return", "checks", "visit", "exit")}
        for key, cell in report["cells"].items()
    }
    return {
        **{k: v for k, v in report.items() if k not in ("fit", "cells")},
        "fit_relevant_rows": [item["relevant"] for item in probe_coverage],
        "cells": cells,
        "relative_path": str(path.relative_to(context.artifacts_dir)),
        "sha256": belief_probe.sha256(path),
    }


def analyze(context, recipe, result):
    outputs = RunArtifacts.from_context(context)
    raw = RunArtifacts(context.artifacts_dir / "analysis", context.artifacts_dir)
    environment = env.RockSampleParams.from_instance(N, K)
    arm = recipe.arms[0]
    spec = ppo.make_spec(environment, d_model=arm.d_model, n_layers=arm.n_layers,
                         n_heads=arm.n_heads, context_len=arm.context_len)
    init = belief_probe.initialization(spec, result["seed_index"], context.seed, recipe.num_keys)
    checkpoints = result["checkpoints"]
    steps = [r["update"] for r in checkpoints]
    if steps != list(recipe.checkpoint_schedule(arm.num_updates(recipe.env_steps_per_seed))):
        raise ValueError("RockSample training did not save the prescribed checkpoints")
    milestone = 1 if context.smoke else cont.SOURCE_UPDATES
    episodes = 16 if context.smoke else 512
    probe_records, steering = [], []
    for checkpoint in [{"update": 0, "env_steps": 0}, *checkpoints]:
        update = checkpoint["update"]
        final = update == steps[-1]
        path = context.artifacts_dir / checkpoint["path"] if update else None
        params = init
        digest = None
        if path:
            with path.open("rb") as stream:
                state = pickle.load(stream)
            if int(state.env_steps) != checkpoint["env_steps"]:
                raise ValueError(f"checkpoint steps differ at update {update}")
            params = jax.device_put(state.params)
            del state
            digest = belief_probe.sha256(path)
        print(f"RockSample checkpoint probes at update {update}", flush=True)
        report = {
            **checkpoint, "checkpoint_sha256": digest,
            "parameter_sha256": checkpoint_beliefs.parameter_hash(params),
            **checkpoint_beliefs.evaluate(spec, environment, params, init, episodes=episodes,
                                          full_battery=(final or update == milestone) and not context.smoke),
        }
        if final:
            report["independent_behavior"] = checkpoint_beliefs.independent_behavior(
                spec, environment, params, episodes=16 if context.smoke else 4096
            )
        artifact = raw.write_json(f"probes_update{update:03d}.json", report)
        record = {
            **checkpoint, "checkpoint_sha256": digest,
            "details": str(artifact.relative_to(context.artifacts_dir)),
            "details_sha256": belief_probe.sha256(artifact),
            "distributions": {
                kind: {
                    "marginals": {name: entry["metrics"] for name, entry in report[kind]["marginals"].items()
                                  if "metrics" in entry},
                    "behavior": report[kind]["behavior"],
                }
                for kind in ("on_policy", "common_checks")
            },
        }
        probe_records.append(record)
        if update in (milestone, steps[-1]):
            confirmation = confirm(context, spec, environment, params, raw, update)
            steering.append({**confirmation, "checkpoint_sha256": digest})
    outputs.write_json("paper_probes.json", {
        "seed": context.seed, "episodes_per_distribution": episodes, "checkpoints": probe_records,
    })
    outputs.write_json("steering_summary.json", {"seed": context.seed, "checkpoints": steering})
