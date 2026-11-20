"""Frozen-policy causal-decoding pilot; no training or cloud provisioning."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
METHODS = ("eq4", "eq4_joint", "policy", "das", "band", "mlp", "null", "norm_null", "intact", "map",
           "context_exact", "context_policy", "context_exact_full", "context_policy_full",
           "action", "behavioural", "behavioural_action")
TESTS = ("ladder", "ladder_full", "moving", "xor_eps02", "xor_v2")


def worker(job):
    test, seed, method, pilot, cpus, beta, write_mode, stage3, campaign, smoke, stage2, diagnose = job
    os.sched_setaffinity(0, set(cpus))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_core as ladder_old
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving as moving_old
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
    from experiments.algorithm_generality_probes_eps02_2026_11.analyze import FINAL, read, restore, write
    from experiments.belief_quotient_causal_decoding_2026_11 import causal, decoders as d
    from experiments.belief_quotient_history_jax_2026_11 import env, models
    from experiments.belief_quotient_replay_edit_2026_11 import replay as lr
    from experiments.belief_quotient_replay_edit_2026_11 import run as lrun
    from experiments.reward_factors_replay_edit_2026_11 import replay as rr, transfer
    from experiments.reward_factors_replay_edit_2026_11 import run as rrun
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment as eps
    from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
    from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process

    jax.config.update("jax_default_matmul_precision", "highest")
    start = time.monotonic()
    ladder = test.startswith("ladder")
    variant = 3 if test == "ladder_full" else 2
    folder = ROOT if not ladder else ROOT.parent / "belief_quotient_causal_decoding_2026_11"
    suffix = "" if beta == 1. and write_mode == "auto" else f"_beta{beta}_{write_mode}"
    prefix = "smoke" if smoke else "campaign" if campaign else "stage3" if stage3 else "diagnosis" if diagnose else "stage2" if stage2 else "pilot" if pilot else "full"
    output = folder / "results" / f"{prefix}_{test}_seed{seed}_{method}{suffix}.json"
    if output.exists():
        existing = json.loads(output.read_text())
        expected_config = dict(test=test, seed=seed, method=method, pilot=pilot, beta=beta,
                               write=write_mode, stage2=stage2)
        if any(existing.get(key) != value for key, value in expected_config.items()):
            raise ValueError(f"incompatible result: {output}")
        print(f"SKIP {output}", flush=True)
        return
    if ladder:
        archived, archive_source = lrun.archive(f"variant_{variant}_transformer/results/belief_steering_v2_seed{seed}_stable.json")
        path = Path("experiments", lrun.STUDY, f"variant_{variant}_transformer", "results", archived["checkpoint"]["run_id"])
        process = eps.ladder_process(variant)
    else:
        cohort = "v2" if test == "xor_v2" else "eps02"
        path, condition, archived, archive_source = rrun.source(cohort, seed)
        process = factor_process(condition, v2 if cohort == "v2" else eps)
    repo = ROOT.parents[1]
    if (repo / path / "resolved_recipe.json").exists():
        recipe = read(repo / path / "resolved_recipe.json")
        locator = path
    else:
        recipe, _ = rrun.archive(path / "resolved_recipe.json")
        manifest, _ = rrun.archive(path / "run_manifest.json")
        locator = (ROOT / "artifacts" / "metadata" / path.name).relative_to(repo)
        write(repo / locator / "run_manifest.json", manifest)
    spec = models.ModelSpec(**recipe["model"])
    params, checkpoint = restore(dict(results_path=str(locator), run_id=path.name, algorithm="reinforce"), FINAL, spec)
    expected = archived["checkpoint"]["sha256"] if ladder else archived["checkpoint_sha256"]
    if checkpoint["sha256"] != expected:
        raise ValueError("checkpoint mismatch")
    for key, value in process._asdict().items():
        np.testing.assert_allclose(value, recipe["process"][key], atol=1e-6)
    policy_logits = lambda h: transfer.logits(params, h)
    policy = lambda h: jax.nn.softmax(policy_logits(h) / 1.5)
    if ladder:
        factor_beliefs = lambda b: b[:, None]
        non_target = lambda p: jnp.ones((len(p), 1))
        non_logits = lambda h: jnp.zeros(1)
    else:
        factor_beliefs = base.factor_beliefs
        non_target = lambda p: p[:, jnp.asarray(rr.INVERSE)].sum(-1)
        non_logits = (lambda h: jax.scipy.special.logsumexp(jax.nn.log_softmax(policy_logits(h) / 1.5)[jnp.asarray(rr.INVERSE)], -1)) if test == "moving" else lambda h: jnp.zeros(1)
    nfit, ntest = (2, 2) if pilot else (24, 12)
    if stage2 and test in ("ladder", "moving"):
        nfit = ntest = 8
    if smoke:
        nfit = ntest = 1
    length, warmup = 1024, 8 if ladder else 64
    def records(random_seed, count):
        if ladder:
            raw = lr.records(spec, process, params, random_seed, count, length)
            return dict(sites=raw["h"], belief=raw["belief"], action=raw["action"])
        return base.collect(spec, params, process, count, length, base._keys(random_seed)[0])
    fit, held = records(42, nfit), records(44, ntest)
    def flatten(x):
        return np.asarray(x[:, warmup:]).reshape(-1, x.shape[-1])
    h = flatten(fit["sites"][:, :, 3])
    ht = flatten(held["sites"][:, :, 3])
    b = np.asarray(factor_beliefs(flatten(fit["belief"])))
    bt = np.asarray(factor_beliefs(flatten(held["belief"])))
    y, yt = b.reshape(len(b), -1), bt.reshape(len(bt), -1)
    affine = d.affine(h, y)
    inverse = d.affine(y, h)
    encoder = jax.jacrev(inverse)(jnp.zeros(y.shape[-1])).T
    band = flatten(d.band_features(fit["sites"][:, :, 3]))
    bandt = flatten(d.band_features(held["sites"][:, :, 3]))
    band_probe = d.affine(band, y)
    probe = d.mlp(h, y) if method == "mlp" else affine
    implied = d.policy_implied(b, np.asarray(policy_logits(jnp.asarray(h)))) if method == "policy" or method.startswith(("context_policy", "behavioural")) else None
    targeted = (lambda p: p[:, jnp.asarray(rr.INVERSE)].sum(-2)) if test == "moving" else lambda p: p
    action_u = jnp.eye(spec.width)
    action_diagnostics = None
    if method in ("action", "behavioural_action"):
        def factor_logits(factor):
            def apply(a):
                z = policy_logits(a)
                return jax.scipy.special.logsumexp(z[jnp.asarray(rr.INVERSE)], axis=1-factor)
            return apply
        if ladder:
            action_u, singular = d.action_basis(h[::16], policy_logits, spec.n_actions)
        else:
            u1, s1 = d.action_basis(h[::16], factor_logits(0), 3)
            u2, s2 = d.action_basis(h[::16], factor_logits(1), 3,
                                    exclude=np.asarray(u1) if test == "moving" else None)
            if test == "moving":
                action_u, singular = u2, s2
            else:
                action_u, _ = d.action_basis(h[::16], lambda a: jnp.concatenate((factor_logits(0)(a), factor_logits(1)(a))), 6)
                singular = dict(factor1=s1, factor2=s2)
        selection = records(43, ntest)
        hs = flatten(selection["sites"][:, :, 3])
        ys = np.asarray(factor_beliefs(flatten(selection["belief"]))).reshape(len(hs), -1)
        ranks, best = {}, None
        for rank in range(1, action_u.shape[1] + 1):
            u = action_u[:, :rank]
            decoder = d.affine(h @ u, y)
            score = float(base.r_squared(decoder(hs @ u), ys))
            ranks[str(rank)] = dict(selection_r2=score, test_r2=float(base.r_squared(decoder(ht @ u), yt)))
            if best is None or score > best[0]:
                best = score, rank, u, decoder
        _, rank, action_u, decoder = best
        action_diagnostics = dict(ranks=ranks, selected_rank=rank, singular_values=singular,
                                  full_test_r2=float(base.r_squared(affine(ht), yt)))
        if method == "action":
            probe = lambda a: decoder(a @ action_u)
            restricted_inverse = d.affine(y, h @ action_u)
            encoder = jax.jacrev(restricted_inverse)(jnp.zeros(y.shape[-1])).T @ action_u.T
    def simplex(raw):
        raw = jnp.clip(raw.reshape(-1, *b.shape[1:]), 0., 1.)
        return raw / jnp.maximum(raw.sum(-1, keepdims=True), 1e-7)
    def decode(activation, past=None):
        if implied is not None:
            return implied(policy_logits(activation)).reshape(len(activation), -1)
        if method == "band":
            return band_probe(jnp.concatenate((activation, past), -1))
        return probe(activation)
    diagnostics = {"affine_r2": base.r_squared(affine(ht), yt),
                   "band_r2": base.r_squared(band_probe(bandt), yt),
                   "decoder_r2": base.r_squared(band_probe(bandt) if method == "band" else probe(ht), yt)}
    if action_diagnostics is not None:
        diagnostics["action_subspace"] = action_diagnostics
    pred = np.asarray(band_probe(bandt) if method == "band" else probe(ht)).reshape(bt.shape)
    diagnostics["rho_r2"] = [base.r_squared(pred[:, f, 2:3], bt[:, f, 2:3]) for f in range(bt.shape[1])]
    diagnostics["delta_r2"] = [base.r_squared((pred[:, f, 0] - pred[:, f, 1])[:, None],
                                              (bt[:, f, 0] - bt[:, f, 1])[:, None]) for f in range(bt.shape[1])]
    if implied is not None:
        decoded = np.asarray(implied(policy_logits(jnp.asarray(ht))))
        actual = policy(jnp.asarray(ht))
        mapped = implied.policy_map(jnp.asarray(bt))
        diagnostics["policy_map_test_kl"] = float(d.kl(actual, mapped).mean())
        diagnostics["policy_map_test_tv"] = float((.5 * jnp.abs(actual - mapped).sum(-1)).mean())
        diagnostics["policy_map_episode_kl"] = np.asarray(d.kl(actual, mapped).reshape(ntest, -1).mean(-1)).tolist()
        diagnostics["policy_map_episode_tv"] = np.asarray((.5 * jnp.abs(actual - mapped).sum(-1)).reshape(ntest, -1).mean(-1)).tolist()
        train_policy = policy(jnp.asarray(h))
        diagnostics["policy_map_train_kl"] = float(d.kl(train_policy, implied.policy_map(jnp.asarray(b))).mean())
        diagnostics["policy_mean_test_kl"] = float(d.kl(actual, train_policy.mean(0)).mean())
        diagnostics["inverse_policy_test_kl"] = float(d.kl(actual, implied.policy_map(jnp.asarray(decoded))).mean())
        diagnostics["policy_implied_mae_per_factor"] = np.abs(decoded - bt).mean((0, 2)).tolist()
        diagnostics["policy_implied_rho_mae"] = np.abs(decoded[..., 2] - bt[..., 2]).mean(0).tolist()
        diagnostics["policy_implied_delta_mae"] = np.abs((decoded[..., 0] - decoded[..., 1]) - (bt[..., 0] - bt[..., 1])).mean(0).tolist()
    basis = jnp.eye(spec.width)
    if method == "das":
        basis, diagnostics["das"] = causal.fit(spec, params, process, ladder, variant, nfit, length,
                                              non_target, beta=beta if test == "moving" else 0.,
                                              frames=(1, 2, 3) if test.startswith("xor") else (2,))
    bank_h, bank_b = jnp.asarray(h[::4]), jnp.asarray(affine(h[::4]))
    if method.startswith("context_"):
        bank_b = (implied(policy_logits(bank_h)).reshape(len(bank_h), -1)
                  if implied is not None else jnp.asarray(y[::4]))
        readout = np.asarray(jax.jacrev(affine)(jnp.zeros(spec.width)))
        readout = readout if ladder or test.startswith("xor") else readout[3:]
        _, singular, axes = np.linalg.svd(readout, full_matrices=False)
        basis = jnp.asarray(axes[singular > singular.max() * d.RCOND].T)
        diagnostics["context_subspace_rank"] = int(basis.shape[1])
    context_scale = jnp.maximum(jnp.asarray(h.std(0)), .01)
    if method in ("policy", "mlp", "band") and write_mode == "donor":
        bank_b = (implied(policy_logits(bank_h)).reshape(len(bank_h), -1) if method == "policy" else
                  band_probe(band[::4]) if method == "band" else probe(bank_h))
        jacobian = jax.jacrev(lambda activation: decode(activation[None], jnp.asarray(band[:1, spec.width:]))[0])(jnp.asarray(h.mean(0)))
        _, singular, axes = np.linalg.svd(np.asarray(jacobian), full_matrices=False)
        basis = jnp.asarray(axes[singular > singular.max() * d.RCOND].T)
    legacy_rows = (dict(sites=np.asarray(fit["sites"][:, warmup:]).reshape(-1, 5, spec.width),
                        belief=flatten(fit["belief"]), action=np.asarray(fit["action"][:, warmup:]).reshape(-1))
                   if pilot or ladder else base.collect_rows(spec, params, process, rows=archived["fit_rows"], seed=archived["fit_seed"]))
    maps = transfer.prepare(legacy_rows) if not ladder else None
    if ladder:
        legacy_maps = ladder_old.fit_maps(dict(h=np.asarray(fit["sites"]).swapaxes(0, 1),
                                               belief=np.asarray(fit["belief"]).swapaxes(0, 1)), min(12000, len(h)))
        null_direction = legacy_maps["orthogonal"][3]
    else:
        null_direction = jnp.asarray(base.null_direction(np.asarray(maps["marginal"]["decoder"]), seed + 9999))
    if not ladder:
        q, schedule = map(jnp.asarray, transfer.task_tables("moving" if test == "moving" else "xor", process))
    else:
        q, schedule = None, None
    if method in ("band", "mlp"):
        sample, past = jnp.asarray(ht[:128]), jnp.asarray(bandt[:128, spec.width:])
        raw = decode(sample, past)
        current = simplex(raw)
        if ladder:
            target = ladder_old.target(current[:, 0], 2).reshape(raw.shape)
        else:
            joint = (current[:, 0, :, None] * current[:, 1, None, :]).reshape(-1, 9)
            request = jnp.asarray(rr.ACTION)[(joint @ q[0]).argmax(-1)]
            anchors = maps["marginal"]["anchors"][jnp.arange(2), request]
            target = jax.nn.one_hot(anchors, 3).reshape(raw.shape)
        dec = lambda a: decode(a, past)
        diagnostics["gradient_checks"] = {}
        for steps in (20, 60, 200):
            delta = jax.jit(lambda a: d.gradient_write(a, target, dec, policy, non_target,
                                                     beta=0., steps=steps))(sample)
            diagnostics["gradient_checks"][str(steps)] = dict(
                initial_mse=float(jnp.mean((raw - target) ** 2)),
                final_mse=float(jnp.mean((dec(sample + delta) - target) ** 2)),
                shift_norm=float(jnp.linalg.norm(delta, axis=-1).mean()),
                policy_tv=float((.5 * jnp.abs(policy(sample + delta) - policy(sample)).sum(-1)).mean()),
                actions_changed=float((policy(sample + delta).argmax(-1) != policy(sample).argmax(-1)).mean()))
    print(f"FIT {test} {method} {time.monotonic()-start:.1f}s", flush=True)

    def rollout(random_seed, episodes, strength, mode, random_actions=False):
        if ladder:
            reset_key, stream = jax.random.split(jax.random.PRNGKey(random_seed))
            step_key = stream
        else:
            reset_key, step_key, _ = base._keys(random_seed)
            stream = step_key
        state = jax.vmap(env.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
        belief = jax.vmap(env.belief_reset, (None, 0))(process, state.token)
        cache = models.init_carry(spec, episodes)
        history = jnp.zeros((episodes, 8, spec.width))
        def body(carry, t):
            state, belief, cache, history, stream = carry
            stream, ak, ek = jax.random.split(stream, 3)
            obs = jax.vmap(env.observe, (None, 0))(process, state)
            z, sites, _ = base.forward(spec, params, cache, obs, t)
            activation = sites[:, 3]
            past = jnp.concatenate((history[:, -1], history[:, -4], history[:, -8]), -1)
            raw = (factor_beliefs(belief).reshape(episodes, -1)
                   if method.startswith(("context_exact", "behavioural")) else decode(activation, past))
            current = simplex(raw)
            if ladder:
                goal = ladder_old.target(current[:, 0], mode)[:, None]
            else:
                joint = (current[:, 0, :, None] * current[:, 1, None, :]).reshape(-1, 9)
                request = jnp.asarray(rr.ACTION)[(joint @ q[t]).argmax(-1)]
                anchors = maps["marginal"]["anchors"][jnp.arange(2), request]
                support = jnp.array([test != "moving", True])
                goal = jnp.where(support[None, :, None], jax.nn.one_hot(anchors, 3), current)
            goal_flat = goal.reshape(episodes, -1)
            delta = (goal_flat - raw) @ encoder
            if method in ("eq4", "eq4_joint", "null", "norm_null"):
                if ladder:
                    delta = ladder_old.ladder_displacement(activation, legacy_maps, 3, mode) @ legacy_maps["encoder"][3]
                else:
                    name = "joint" if method == "eq4_joint" else "marginal"
                    mapping = maps[name]
                    legacy_raw = activation @ mapping["decoder"] + mapping["bias"]
                    legacy_goal = moving_old.belief_target(legacy_raw, name, q[t], mapping["anchors"], (test != "moving", True))
                    delta = (legacy_goal - legacy_raw) @ mapping["embedding"]
            use_write = write_mode if write_mode != "auto" else ("donor" if method == "das" else "gradient" if method in ("mlp", "band", "policy") else "null" if method == "null" else "eq4")
            supported = jnp.zeros(episodes, bool)
            if method.startswith("behavioural"):
                delta = d.behavioural_write(activation, implied.policy_map(goal), policy, action_u,
                                             targeted, non_target, beta=beta if test == "moving" else 0.)
            elif method.startswith("context_"):
                preserve = jnp.zeros(y.shape[-1]) if ladder or test.startswith("xor") else jnp.array([1., 1., 1., 0., 0., 0.])
                delta, supported = d.contextual_donor(activation, goal_flat, raw, bank_h, bank_b,
                                                     basis, preserve, context_scale, full=method.endswith("_full"))
            elif use_write == "gradient":
                decoder = lambda a: decode(a, past)
                delta = d.gradient_write(activation, goal_flat, decoder, policy, non_target,
                                         beta=0. if ladder or test.startswith("xor") else beta, steps=60 if stage2 else 20 if pilot else 60)
            elif use_write == "donor":
                preserve = jnp.zeros(y.shape[-1]) if ladder or test.startswith("xor") else jnp.array([1., 1., 1., 0., 0., 0.])
                delta, _ = d.donor_write(activation, goal_flat, raw, bank_h, bank_b, basis, preserve)
            elif use_write == "null":
                delta = d.null_write(activation, delta, non_logits)
            elif method == "norm_null":
                delta = jnp.linalg.norm(delta, axis=-1, keepdims=True) * null_direction
            elif method in ("intact", "map"):
                delta = jnp.zeros_like(delta)
            shift = lambda a: strength * delta
            written, _, cache = base.forward(spec, params, cache, obs, t, site=3, shift=shift)
            action = written.argmax(-1)
            if method == "map" and not ladder:
                action = (belief @ q[t]).argmax(-1)
            if random_actions:
                action = jax.random.randint(ak, (episodes,), 0, spec.n_actions)
            nxt, reward, _ = jax.vmap(env.step, (None, 0, 0, 0))(
                process, jax.random.split(ek if ladder else jax.random.fold_in(step_key, t), episodes), state, action)
            if ladder:
                score = (state.state == 2).astype(float)
                factor1 = jnp.zeros_like(score)
                changed = action != z.argmax(-1)
            else:
                score = schedule[t, :, state.state].sum(-1)
                factors = jnp.asarray(rr.ACTION)
                # Factor reward uses the pre-transition hidden joint state.
                factor1 = (state.state // 3 == 2).astype(float)
                changed = factors[action, 0] != factors[z.argmax(-1), 0]
            belief = jax.vmap(env.belief_update, (None, 0, 0, 0))(process, belief, action, nxt.token)
            history = jnp.concatenate((history[:, 1:], activation[:, None]), 1)
            return (nxt, belief, cache, history, stream), jnp.stack((score, factor1, changed, supported), -1)
        _, result = jax.lax.scan(body, (state, belief, cache, history, stream), jnp.arange(length))
        return result.mean((0, 1))
    episodes = 2 if pilot else (32 if ladder else 256)
    tune_episodes = 2 if pilot else (16 if ladder else 64)
    if stage2 and test in ("ladder", "moving"):
        episodes = tune_episodes = 8
    if campaign:
        episodes, tune_episodes = 32, 16
    if smoke:
        episodes = tune_episodes = 1
    result = {}
    for mode in ((2, 1) if ladder else (0,)):
        evaluate = jax.jit(lambda strength, random_seed: rollout(random_seed, episodes, strength, mode))
        tune_seed = 1042 if ladder else archived["tune_seed"]
        confirm_seed = 2042 if ladder else archived["confirm_seed"]
        tune = jax.jit(lambda strength: rollout(tune_seed, tune_episodes, strength, mode))
        grid = ladder_old.LAMBDAS if ladder else base.STRENGTHS
        if smoke:
            grid = (0., 1.)
        if ladder and mode == 1:
            strength, curve = 1., {}
        elif method in ("intact", "map"):
            strength, curve = 0., {}
        elif method in ("eq4", "eq4_joint", "norm_null"):
            if ladder:
                strength = next(r["strength"] for r in archived["confirmation"]
                                if r.get("site") == 3 and r["condition"] == "selected_rho_flip")
            else:
                target_name = "joint" if method == "eq4_joint" else "marginal"
                task_name = "moving" if test == "moving" else "xor"
                strength = archived["rewards"][task_name]["confirmation"]["comparisons"][f"discounted_{target_name}_block_4"]["strength"]
            curve = {}
        else:
            curve = {str(s): float(tune(jnp.float32(s))[0]) for s in grid}
            if ladder:
                random_floor = float(rollout(tune_seed, tune_episodes, jnp.float32(0.), mode, True)[0])
                strength = next((s for s in grid if curve[str(s)] <= random_floor), min(grid, key=lambda s: curve[str(s)]))
            else:
                strength = min(grid, key=lambda s: -curve[str(s)])
        values = np.asarray(evaluate(jnp.float32(strength), jnp.int32(confirm_seed)))
        result[str(mode)] = dict(strength=strength, tuning=curve, reward=float(values[0]),
                                factor1_reward=float(values[1]), actions_changed=float(values[2]), donor_coverage=float(values[3]))
    report = dict(test=test, method=method, seed=seed, pilot=pilot, beta=beta, write=write_mode, checkpoint=checkpoint,
                  archive=archive_source, diagnostics=diagnostics, metrics=result,
                  wall_seconds=time.monotonic()-start, cpus=cpus, fit_episodes=nfit,
                  confirmation_episodes=episodes, horizon=length,
                  tuning_episodes=tune_episodes, stage2=stage2,
                  stage3=stage3, campaign=campaign,
                  smoke=smoke,
                  upper_bound_style_reference=method.startswith("behavioural"),
                  revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
    partial = output.with_suffix(".partial")
    write(partial, report)
    partial.replace(output)
    print(f"DONE {test} {method} {report['wall_seconds']:.1f}s {json.dumps(result)}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--test", nargs="+", choices=TESTS, default=TESTS)
    parser.add_argument("--method", nargs="+", choices=(*METHODS, "all"), default=("all",))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--stage3", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cpus", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--write", choices=("auto", "donor", "gradient", "null"), default="auto")
    args = parser.parse_args()
    if args.stage3:
        args.stage2 = True
    if args.stage2:
        args.pilot = True
    if args.campaign and args.pilot:
        parser.error("--campaign cannot be combined with pilot stages")
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cpus = sorted(os.sched_getaffinity(0))[:args.max_cpus]
    workers = min(args.workers, len(cpus))
    methods = METHODS if "all" in args.method else tuple(dict.fromkeys(args.method))
    jobs = [(t, args.seed, m, args.pilot, cpus[i % workers::workers], args.beta, args.write, args.stage3, args.campaign, args.smoke, args.stage2, args.diagnose)
            for i, (t, m) in enumerate((t, m) for t in args.test for m in methods)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, jobs))


if __name__ == "__main__":
    main()
