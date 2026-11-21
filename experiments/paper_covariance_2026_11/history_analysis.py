"""Manuscript geometry controls and steering on locally produced checkpoints."""

import hashlib
import json

import jax
import numpy as np
from harness.artifacts import RunArtifacts
from harness.seeding import child_seed_sequence, seed_sequence_to_int

from analysis.probes.controls import fit_grouped_affine, score_prediction
from experiments.algorithm_generality_probes_eps02_2026_11 import (
    analyze as ladder_probe,
    paper_coarse,
    paper_probe,
    steer_ladder,
    steer_moving,
    steer_moving_base,
    steer_storage,
)
from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.reward_factors_history_jax_2026_11.process import factor_targets
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11 import (
    shared as factors,
)


def load_actor(path, spec):
    with np.load(path, allow_pickle=False) as saved:
        template = models.init_params(spec, jax.random.PRNGKey(0))
        return ladder_probe.actor_arrays(saved, template, "reinforce")


def modified_history(observations, kind, seed, lengths):
    """Paper's paired-token/action controls on complete greedy prefixes."""
    rng = np.random.default_rng(seed)
    result = observations.copy()
    if kind == "shuffle_control":
        for index, length in enumerate(lengths):
            result[index, 1:length] = observations[
                index, rng.permutation(np.arange(1, length))
            ]
        return result
    valid = np.arange(observations.shape[1])[None, :] < lengths[:, None]
    valid[:, 0] = False
    reset, pool = observations[:, 0], observations[valid]
    result[:, 0] = reset[rng.integers(len(reset), size=len(reset))]
    result[valid] = pool[rng.integers(len(pool), size=len(pool))]
    return result


def frozen_controls(context, spec, process, params, *, site, length):
    _, rollout, replay = paper_probe.runtime()
    warmup, rows = 128, 4_000 if context.smoke else 20_000
    usable = length - warmup
    episodes = -(-rows // usable)
    lengths = np.full(episodes, length)
    lengths[-1] = warmup + rows - (episodes - 1) * usable
    groups = np.repeat(np.arange(episodes), usable)[:rows]

    def seed_at(*path):
        return seed_sequence_to_int(child_seed_sequence(context.seed, path))

    histories = [
        rollout(spec, process, params, jax.random.PRNGKey(seed_at(key)), episodes, length)
        for key in (700, 701)
    ]
    beliefs = [paper_probe.flatten(item["belief"], warmup, rows) for item in histories]
    if site == "embedding":
        targets = [{"belief": belief} for belief in beliefs]
    else:
        targets = [factor_targets(belief) for belief in beliefs]
        targets = [{name: value for name, value in t.items() if name != "joint"} for t in targets]
    reports = {}
    for index, name in enumerate(("shuffle_control", "random_input_control")):
        features = []
        for item, streams in zip(histories, ((index,), (index, 1))):
            modified = modified_history(np.asarray(item["obs"]), name, seed_at(*streams), lengths)
            features.append(paper_probe.flatten(replay(spec, params, modified, site), warmup, rows))
        report = {}
        for target, fit_y in targets[0].items():
            offset = index + 1 if target == "belief" else int(target[-1])
            weight, bias, fit = fit_grouped_affine(features[0], fit_y, groups,
                                                   seed=context.seed + offset)
            report[target] = {
                **score_prediction(features[1] @ weight + bias, targets[1][target]),
                "fit": fit,
            }
        reports[name] = report
    return reports


def ladder_geometry(context, spec, process, params, init, variant, length, *, final):
    _, rollout, replay = paper_probe.runtime()
    fit_rows, test_rows = (960, 1920) if context.smoke else (60_000, 80_000)
    counts, warmup = (fit_rows, test_rows), 64
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(context.seed), 2))
    raw = [
        rollout(spec, process, params, key, -(-count // (length - warmup)), length)
        for key, count in zip(keys, counts)
    ]
    features = [paper_probe.flatten(r["embedding"], warmup, n) for r, n in zip(raw, counts)]
    beliefs = [paper_probe.flatten(r["belief"], warmup, n) for r, n in zip(raw, counts)]
    targets = [dict(belief=b, rho=b[:, 2:3], delta=b[:, :1] - b[:, 1:2]) for b in beliefs]
    if variant in (2, 3):
        process_arrays = process._asdict()
        for r, n, target in zip(raw, counts, targets):
            merged = paper_coarse.merged_filter(np.asarray(r["obs"]), process_arrays)[..., 2:3]
            if variant == 2:
                exact = paper_probe.coarse_filter(np.asarray(r["obs"]), process_arrays)
                np.testing.assert_allclose(exact[..., None], merged, atol=1e-12, rtol=0)
            name = "coarse_b" if variant == 2 else "merged_token_rho"
            target[name] = paper_probe.flatten(merged, warmup, n)
    metrics = {
        "fit_rows": fit_rows, "test_rows": test_rows, "warmup": warmup,
        "scalars": {name: paper_probe.fit_score(*features, y, targets[1][name])[0]
                    for name, y in targets[0].items()},
    }
    if final:
        classes = [paper_probe.flatten(r["action"], warmup, n) for r, n in zip(raw, counts)]
        matched = [
            paper_probe.flatten(replay(spec, init, r["obs"], "embedding"), warmup, n)
            for r, n in zip(raw, counts)
        ]
        metrics["within_class"] = {
            "trained": ladder_probe.within(params, features, beliefs, classes),
            "matched_initialization": ladder_probe.within(init, matched, beliefs, classes),
        }
    return metrics


def analyze_checkpoints(context, summary, variant=None):
    outputs = RunArtifacts.from_context(context)
    recipe = json.loads((context.results_dir / "resolved_recipe.json").read_text())
    spec = models.ModelSpec(**recipe["model"])
    process = env.make_process(**recipe["process"])
    length = recipe["training"]["episode_length"]
    steps = [item["checkpoint_step"] for item in summary["checkpoint_reports"]]
    if steps != sorted(set(steps)) or len(steps) < 2 or steps[0] != 0:
        raise ValueError("expected strictly increasing probes starting at initialization")
    indices = sorted({*range(min(6, len(steps))), *range(10, len(steps), 5), len(steps) - 1})
    directory = context.artifacts_dir / "checkpoints"
    init = load_actor(directory / "step_0000000000.npz", spec)
    records = []
    with jax.default_matmul_precision("highest"):
        for index in indices:
            step = steps[index]
            path = directory / f"step_{step:010d}.npz"
            params = load_actor(path, spec)
            record = dict(step=step, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            site = "embedding" if variant else "residual"
            record["controls"] = frozen_controls(context, spec, process, params, site=site, length=length)
            if variant:
                record.update(ladder_geometry(context, spec, process, params, init, variant, length,
                                              final=index == len(steps) - 1))
            records.append(record)
            print(f"additional probes at {step}", flush=True)
    outputs.write_json("paper_probes.json", {"seed": context.seed, "checkpoints": records})
    return spec, process, params, path


def save_steering(context, raw_dir, report, compact):
    raw_path = RunArtifacts(raw_dir, context.artifacts_dir).write_json(
        "full_report.json", report
    )
    compact["full_report"] = {
        "relative_path": str(raw_path.relative_to(context.artifacts_dir)),
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    RunArtifacts.from_context(context).write_json("steering_summary.json", compact)


def analyze_ladder(context, variant, summary):
    spec, process, params, checkpoint = analyze_checkpoints(context, summary, variant)
    raw = context.artifacts_dir / "steering"
    raw.mkdir(parents=True, exist_ok=True)
    with jax.default_matmul_precision("highest"):
        report = steer_ladder.run(spec, process, params, raw, variant, context.seed,
                                 context.smoke, environment)
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    compact = steer_storage.compact(report, None)
    compact["checkpoint_sha256"] = report["checkpoint_sha256"]
    save_steering(context, raw, report, compact)


def analyze_factors(context, condition, summary):
    spec, process, params, checkpoint = analyze_checkpoints(context, summary)
    if condition != "reward_both":
        return
    raw = context.artifacts_dir / "steering"
    source = dict(study=factors.STUDY, condition=condition, results=context.results_dir,
                  run_id=context.run_id, training_seed=context.seed)

    def actor(path, requested_spec):
        if path != checkpoint or requested_spec != spec:
            raise ValueError("moving-reward driver selected a different checkpoint")
        return params

    with jax.default_matmul_precision("highest"):
        report = steer_moving.analyze(
            source, checkpoint, seed=42, tune_episodes=2 if context.smoke else 64,
            confirm_episodes=2 if context.smoke else 256, load_actor=actor,
            fit_rows=256 if context.smoke else None, process=process,
        )
        fit_n, test_n = (256, 512) if context.smoke else (60_000, 80_000)
        fit = steer_moving_base.collect_rows(spec, params, process, rows=fit_n, seed=42)
        test = steer_moving_base.collect_rows(spec, params, process, rows=test_n, seed=43)
        fit_targets = steer_moving_base.map_targets(fit["belief"])
        test_targets = steer_moving_base.map_targets(test["belief"])
        sites = [f"block_{i + 1}" for i in range(spec.n_layers)] + ["post_final_norm"]
        probe = {}
        for i, site in enumerate(sites):
            features = fit["sites"][:, i].astype(np.float64)
            probe[site] = {}
            for name in steer_moving.TARGETS:
                weight, bias = steer_moving_base.affine_fit(features, fit_targets[name].astype(np.float64))
                probe[site][name] = steer_moving_base.r_squared(
                    test["sites"][:, i] @ weight + bias, test_targets[name]
                )
        report["held_out_probe_r_squared_by_site"] = probe
    controls = report["rewards"]["moving"]
    compact = {
        "checkpoint_sha256": report["checkpoint_sha256"],
        "fit_seed": report["fit_seed"], "tune_seed": report["tune_seed"],
        "confirm_seed": report["confirm_seed"], "confirm_episodes": report["confirm_episodes"],
        "confirmation": controls["confirmation"],
        "selected_strength": controls["strength_tuning"]["selected_strength"],
        "held_out_probe_r_squared_by_site": probe,
    }
    save_steering(context, raw, report, compact)
