"""Site-specific exact/decoded affine/MLP encoders on frozen checkpoints."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TESTS = ("ladder", "ladder_full", "moving", "xor_eps02", "xor_v2")
ARMS = tuple(itertools.product(("affine", "mlp"), ("exact", "decoded")))


def initialize(cpu_queue):
    os.sched_setaffinity(0, set(cpu_queue.get()))


def folder(test):
    return ROOT if not test.startswith("ladder") else ROOT.parent / "belief_quotient_block2_steering_2026_11"


def result_path(test, seed, representation, kind, source, site, smoke=False):
    return folder(test) / "results" / f"{'smoke' if smoke else 'pilot'}_site{site}_{test}_seed{seed}_{representation}_{kind}_{source}.json"


def choose_strength(grid, scores, ladder, random_floor=None):
    if ladder:
        return next((s for s in grid if scores[str(s)] <= random_floor), min(grid, key=lambda s: scores[str(s)]))
    return min(grid, key=lambda s: -scores[str(s)])


def worker(job):
    test, seed, site, smoke = job
    pilot = True
    cpus = sorted(os.sched_getaffinity(0))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_core as ladder_old
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving as moving_old
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
    from experiments.algorithm_generality_probes_eps02_2026_11.analyze import FINAL, read, restore, write
    from experiments.belief_quotient_encoder_nonlinearity_2026_11 import encoders
    from experiments.belief_quotient_history_jax_2026_11 import env, models
    from experiments.belief_quotient_replay_edit_2026_11 import replay as lr, run as lrun
    from experiments.reward_factors_replay_edit_2026_11 import replay as rr, transfer, run as rrun
    from .run import prepare
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment as eps
    from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
    from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process

    jax.config.update("jax_default_matmul_precision", "highest")
    start = time.monotonic()
    ladder = test.startswith("ladder")
    representations = ("marginal",) if ladder else ("marginal", "joint")
    if all((p := result_path(test, seed, r, k, s, site, smoke)).exists()
           and json.loads(p.read_text()).get("action_change_reference") == "intact_same_executed_history"
           for r in representations for k, s in ARMS):
        print(f"SKIP {test} seed{seed}", flush=True)
        return
    variant = 3 if test == "ladder_full" else 2
    if ladder:
        archived, archive_source = lrun.archive(f"variant_{variant}_transformer/results/belief_steering_v2_seed{seed}_stable.json")
        path = Path("experiments", lrun.STUDY, f"variant_{variant}_transformer", "results", archived["checkpoint"]["run_id"])
        process = eps.ladder_process(variant)
    else:
        cohort = "v2" if test == "xor_v2" else "eps02"
        path, condition, archived, archive_source = rrun.source(cohort, seed)
        process = factor_process(condition, v2 if cohort == "v2" else eps)
    if (REPO / path / "resolved_recipe.json").exists():
        recipe = read(REPO / path / "resolved_recipe.json")
        locator = path
    else:
        recipe, _ = rrun.archive(path / "resolved_recipe.json")
        manifest, _ = rrun.archive(path / "run_manifest.json")
        locator = (ROOT / "artifacts" / "metadata" / path.name).relative_to(REPO)
        write(REPO / locator / "run_manifest.json", manifest)
    spec = models.ModelSpec(**recipe["model"])
    params, checkpoint = restore(dict(results_path=str(locator), run_id=path.name, algorithm="reinforce"), FINAL, spec)
    expected = archived["checkpoint"]["sha256"] if ladder else archived["checkpoint_sha256"]
    if checkpoint["sha256"] != expected:
        raise ValueError("checkpoint mismatch")
    if spec.kind != "transformer" or spec.n_layers != 4:
        raise ValueError("expected frozen four-block transformer")
    for key, value in process._asdict().items():
        np.testing.assert_allclose(value, recipe["process"][key], atol=1e-6)
    count, held_count = (1, 1) if smoke else (24, 12)
    warmup = 8 if ladder else 64

    def records(random_seed, episodes):
        if ladder:
            raw = lr.records(spec, process, params, random_seed, episodes, 1024)
            raw["sites"] = raw["h"]
        else:
            raw = base.collect(spec, params, process, episodes, 1024, base._keys(random_seed)[0])
        return {key: np.asarray(raw[key])[:, warmup:].reshape(-1, *raw[key].shape[2:])
                for key in ("sites", "belief", "action")}

    fit = records(42, count) if ladder or smoke else base.collect_rows(
        spec, params, process, rows=archived["fit_rows"], seed=archived["fit_seed"])
    if ladder:
        fit = {key: value[:12000] for key, value in fit.items()}
    validation, held = records(43, held_count), records(44, held_count)
    maps = prepare(fit, site) if not ladder else None
    q, schedule = (None, None) if ladder else map(jnp.asarray, transfer.task_tables("moving" if test == "moving" else "xor", process))
    h, hv, ht = (np.asarray(rows["sites"][:, site]) for rows in (fit, validation, held))
    print(f"DATA {test} seed{seed}: fit={len(h)} validation={len(hv)} test={len(ht)} {time.monotonic()-start:.1f}s", flush=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__), Path(encoders.__file__))}

    for representation in representations:
        rcond = ladder_old.AFFINE_RCOND if ladder else 1e-10
        exact, exactv, exactt = (np.asarray(rows["belief"][:, :2]) if ladder else base.map_targets(rows["belief"])[representation]
                                for rows in (fit, validation, held))
        if ladder:
            decoder, bias = map(jnp.asarray, ladder_old.affine(h, exact))
            anchors = None
            historical_strength = next(r["strength"] for r in archived["confirmation"]
                                       if r.get("site") == 3 and r["condition"] == "selected_rho_flip")
        else:
            mapping = maps[representation]
            decoder, bias, anchors = (mapping[key] for key in ("decoder", "bias", "anchors"))
            task = "moving" if test == "moving" else "xor"
            historical_strength = archived["rewards"][task]["confirmation"]["comparisons"][f"discounted_{representation}_block_4"]["strength"]
        decoded, decodedv, decodedt = (np.asarray(jnp.asarray(a) @ decoder + bias) for a in (h, hv, ht))
        for kind, source in ARMS:
            output = result_path(test, seed, representation, kind, source, site, smoke)
            if output.exists() and json.loads(output.read_text()).get("action_change_reference") == "intact_same_executed_history":
                print(f"SKIP {output.name}", flush=True)
                continue
            arm_start = time.monotonic()
            features, val_features, test_features = ((exact, exactv, exactt) if source == "exact" else (decoded, decodedv, decodedt))
            encoder = encoders.fit(features, h, val_features, hv, kind, rcond=rcond)
            diagnostics = dict(encoder.diagnostics, test_r2=encoders.r_squared(encoder(jnp.asarray(test_features)), ht),
                               validation_r2=encoders.r_squared(encoder(jnp.asarray(val_features)), hv))
            if kind == "affine" and source == "exact" and not ladder:
                np.testing.assert_allclose(encoder.params[0], maps[representation]["embedding"], atol=0, rtol=0)
            fit_seconds = time.monotonic() - arm_start

            def rollout(random_seed, episodes, strength, mode, random_actions=False):
                if ladder:
                    reset_key, stream = jax.random.split(jax.random.PRNGKey(random_seed))
                    step_key = stream
                else:
                    reset_key, step_key, _ = base._keys(random_seed)
                    stream = step_key
                state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
                cache = models.init_carry(spec, episodes)

                def body(carry, t):
                    state, cache, shadow_cache, stream = carry
                    stream, ak, ek = jax.random.split(stream, 3)
                    obs = jax.vmap(env.observe, (None, 0))(process, state)
                    intact, sites, _ = base.forward(spec, params, cache, obs, t)
                    shadow, _, shadow_cache = base.forward(spec, params, shadow_cache, obs, t)
                    current = sites[:, site] @ decoder + bias
                    if ladder:
                        belief = jnp.concatenate((current, 1-current.sum(-1, keepdims=True)), -1)
                        goal = ladder_old.target(belief, mode)[:, :2]
                    else:
                        goal = moving_old.belief_target(current, representation, q[t], anchors, (test != "moving", True))
                    delta = encoder.displacement(current, goal)
                    written, _, cache = base.forward(spec, params, cache, obs, t, site=site, shift=lambda a: strength * delta)
                    action = written.argmax(-1)
                    if random_actions:
                        action = jax.random.randint(ak, (episodes,), 0, spec.n_actions)
                    nxt, _, _ = jax.vmap(env.step, (None, 0, 0, 0))(
                        process, jax.random.split(ek if ladder else jax.random.fold_in(step_key, t), episodes), state, action)
                    score = (state.state == 2).astype(float) if ladder else schedule[t, :, state.state].sum(-1)
                    factor1 = jnp.zeros_like(score) if ladder else (state.state // 3 == 2).astype(float)
                    changed = action != shadow.argmax(-1) if ladder else jnp.asarray(rr.ACTION)[action, 0] != jnp.asarray(rr.ACTION)[shadow.argmax(-1), 0]
                    local_changed = action != intact.argmax(-1) if ladder else jnp.asarray(rr.ACTION)[action, 0] != jnp.asarray(rr.ACTION)[intact.argmax(-1), 0]
                    return (nxt, cache, shadow_cache, stream), jnp.stack((score, factor1, changed, local_changed), -1)
                _, values = jax.lax.scan(body, (state, cache, cache, stream), jnp.arange(1024))
                return values.mean((0, 1))

            tuning_episodes, confirmation_episodes = (1, 1) if smoke else (16, 32)
            grid = ladder_old.LAMBDAS if ladder else base.STRENGTHS
            if smoke:
                grid = (0., 1.)
            tune_seed, confirm_seed = (1042, 2042) if ladder else (archived["tune_seed"], archived["confirm_seed"])
            metrics, matched_metrics = {}, {}
            for mode in ((2, 1) if ladder else (0,)):
                tune = jax.jit(lambda strength: rollout(tune_seed, tuning_episodes, strength, mode))
                confirm = jax.jit(lambda strength: rollout(confirm_seed, confirmation_episodes, strength, mode))
                scores = {} if ladder and mode == 1 else {str(s): float(tune(jnp.float32(s))[0]) for s in grid}
                random_floor = float(rollout(tune_seed, tuning_episodes, jnp.float32(0.), mode, True)[0]) if ladder and mode == 2 else None
                selected = 1. if ladder and mode == 1 else choose_strength(grid, scores, ladder, random_floor)
                fixed = 1. if ladder and mode == 1 else historical_strength

                def summarize(strength):
                    values = np.asarray(confirm(jnp.float32(strength)))
                    return dict(strength=strength, reward=float(values[0]), factor1_reward=float(values[1]), actions_changed=float(values[2]),
                                local_actions_changed=float(values[3]))

                matched_metrics[str(mode)] = summarize(fixed)
                metrics[str(mode)] = dict(matched_metrics[str(mode)] if selected == fixed else summarize(selected),
                                          tuning=scores, random_floor=random_floor)
            report = dict(test=test, seed=seed, representation=representation, encoder=kind, source=source,
                          historical_encoder_source="exact", write_current_source="probe_decoded_unclipped",
                          pilot=pilot, checkpoint=checkpoint, archive=archive_source, site=site, smoke=smoke,
                          positions="latest_every_step", action_change_reference="intact_same_executed_history",
                          train_rows=len(h), validation_rows=len(hv), test_rows=len(ht),
                          data_seeds=dict(fit=42, validation=43, test=44, tuning=tune_seed, confirmation=confirm_seed),
                          fit_episodes=count if ladder or smoke else None, held_episodes=held_count,
                          tuning_episodes=tuning_episodes, confirmation_episodes=confirmation_episodes, horizon=1024,
                          diagnostics=diagnostics, metrics=metrics, matched_lambda_metrics=matched_metrics,
                          fit_seconds=fit_seconds, arm_seconds=time.monotonic()-arm_start,
                          worker_seconds=time.monotonic()-start, cpus=list(cpus), revision=revision, code_hashes=code_hashes)
            partial = output.with_suffix(".partial")
            write(partial, report)
            partial.replace(output)
            print(f"DONE {test} seed{seed} {representation} {kind}/{source}: R2={diagnostics['test_r2']:.4f} {report['arm_seconds']:.1f}s {json.dumps(metrics)}", flush=True)
            jax.clear_caches()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--test", nargs="+", choices=TESTS, default=TESTS)
    parser.add_argument("--seed", nargs="+", type=int, default=(42,))
    parser.add_argument("--site", type=int, choices=(1, 3), nargs="+", default=(1,))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    workers = min(args.workers, len(available))
    if workers < 1:
        parser.error("positive worker count required")
    jobs = [(test, seed, site, args.smoke) for seed, test, site in itertools.product(args.seed, args.test, args.site)]
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cpu_queue = multiprocessing.Queue()
    for i in range(workers):
        cpu_queue.put(available[i::workers])
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(cpu_queue,)) as pool:
        list(pool.map(worker, jobs))


if __name__ == "__main__":
    main()
