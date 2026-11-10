"""Offline marginal-belief steering of the three upper-plateau r7 policies."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_probe as bp
from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm

AGENTS = [("20261002T191712Z-7f1dc1ef", 3),
          ("20261002T182024Z-ad0c3a34", 2),
          ("20261002T191712Z-7f1dc1ef", 2)]
SEEDS = dict(fit=41101, test=51203, tune=61307, confirm=71411, null=81517)
GRID = [.5, 1, 1.5, 2, 3]
VARIANTS = ("encoder", "calibrated_encoder", "min_norm_decoder")
CONFIRM_SEED = 91423


def affine(x, y):
    design = np.column_stack((x, np.ones(len(x))))
    coef = np.linalg.lstsq(design, y, rcond=1e-6)[0]
    return coef[:-1], coef[-1]


@partial(jax.jit, static_argnames=("spec",))
def replay(spec, params, observations):
    def one(sequence):
        def advance(cache, obs):
            sites = []
            def capture(site, x):
                sites.append(x.reshape(-1))
                return x
            _, cache = tm.encode_step(spec, params, cache, obs,
                                      residual_transform=capture)
            return cache, jnp.stack(sites)
        return jax.lax.scan(advance, tm.empty_cache(spec), sequence)[1]
    return jax.vmap(one)(observations)


def maps(spec, env, params, episodes):
    streams = []
    for seed in (SEEDS["fit"], SEEDS["test"]):
        rows = bp.collect(spec, env, params, params, episodes=episodes, key_seed=seed)
        targets = bp.targets(rows, env).marginals
        features = np.asarray(replay(spec, params, rows["obs"]))[rows["valid"]]
        streams.append((features.astype(np.float64), targets))
    (x, y), (xt, yt) = streams
    decoders, offsets, encoders, scores, nulls = [], [], [], [], []
    rng = np.random.default_rng(SEEDS["null"])
    for site in range(spec.n_layers + 1):
        d, b = affine(x[:, site], y)
        e, _ = affine(y, x[:, site])
        residual = yt - (xt[:, site] @ d + b)
        scores.append((1 - (residual**2).sum(0) /
                       ((yt - yt.mean(0))**2).sum(0)).tolist())
        u, s, _ = np.linalg.svd(d, full_matrices=False)
        basis = u[:, s > s[0] * 1e-6]
        random = rng.normal(size=(env.k, spec.d_model))
        random -= (random @ basis) @ basis.T
        random /= np.linalg.norm(random, axis=1, keepdims=True)
        decoders.append(d); offsets.append(b); encoders.append(e); nulls.append(random)
    return tuple(jnp.asarray(v, dtype=jnp.float32) for v in
                 (decoders, offsets, encoders, nulls)), scores


def shift(h, d, b, e, null, rock, target, strength, control, active):
    before = h @ d + b
    delta = strength * (target - before[rock]) * e[rock] * active
    delta = jnp.where(control, jnp.linalg.norm(delta) * null[rock], delta)
    after = (h + delta) @ d + b
    return h + delta, before[rock], after[rock]


def calibrated_maps(fitted):
    d, b, e, null = map(np.asarray, fitted)
    calibrated, minimum, diagnostics = [], [], []
    for w, encoder in zip(d.astype(float), e.astype(float)):
        m = encoder @ w
        if np.linalg.matrix_rank(m) != m.shape[0] or np.linalg.matrix_rank(w) != w.shape[1]:
            raise ValueError("endpoint calibration requires full-rank decoder and encoder product")
        ce = np.linalg.solve(m, encoder)
        mn = np.linalg.pinv(w, rcond=1e-10)
        ce32, mn32 = ce.astype(np.float32), mn.astype(np.float32)
        diagnostics.append(dict(encoder_product_condition=float(np.linalg.cond(m)),
            calibrated_identity_max_error=float(np.max(np.abs(ce32 @ w.astype(np.float32)-np.eye(7)))),
            minimum_identity_max_error=float(np.max(np.abs(mn32 @ w.astype(np.float32)-np.eye(7)))),
            encoder_row_norms=np.linalg.norm(encoder, axis=1).tolist(),
            calibrated_row_norms=np.linalg.norm(ce32, axis=1).tolist(),
            minimum_row_norms=np.linalg.norm(mn32, axis=1).tolist()))
        calibrated.append(ce32); minimum.append(mn32)
    return {name: (jnp.asarray(d), jnp.asarray(b), jnp.asarray(encoder), jnp.asarray(null))
            for name, encoder in zip(VARIANTS, (e, calibrated, minimum))}, diagnostics


@partial(jax.jit, static_argnames=("spec", "env", "episodes"))
def rollout(spec, env, params, fitted, *, episodes, seed, rock, good,
            site, strength, control):
    d, b, e, null = fitted
    obs, state = jax.vmap(partial(rs.reset, env))(
        jax.random.split(jax.random.key(seed), episodes))
    state = state._replace(qualities=state.qualities.at[:, rock].set(jnp.asarray(good, bool)))
    initial_good = state.qualities
    cache = jax.vmap(lambda _: tm.empty_cache(spec))(jnp.arange(episodes))
    def advance(carry, t):
        obs, state, cache, done, sampled, key = carry
        active = ~sampled[:, rock] | jnp.asarray(good, bool)
        def encode(c, o, enabled):
            diagnostics = []
            def transform(index, h):
                changed, before, after = shift(h.reshape(-1), d[index], b[index],
                    e[index], null[index], rock, 1 - good, strength, control, enabled)
                decoded_delta = (changed-h.reshape(-1)) @ d[index]
                other_delta = decoded_delta.at[rock].set(0)
                diagnostics.append(jnp.stack((before, after,
                    jnp.max(jnp.abs(other_delta)), jnp.abs(after-(1-good)),
                    jnp.linalg.norm(changed-h.reshape(-1)))))
                return jnp.where(index == site, changed.reshape(h.shape), h)
            z, c = tm.encode_step(spec, params, c, o, residual_transform=transform)
            return z, c, jnp.stack(diagnostics)[site]
        z, cache, diagnostic = jax.vmap(encode)(cache, obs, active)
        key, ak, ek = jax.random.split(key, 3)
        logits, _ = tm.heads(params, z)
        action = jax.random.categorical(ak, logits).astype(jnp.int32)
        out = jax.vmap(partial(rs.step, env))(
            jax.random.split(ek, episodes), state, action)
        picked = ((action == 4)[:, None] &
                  jnp.all(state.rover[:, None] == jnp.asarray(env.rocks), axis=-1)
                  & ~done[:, None])
        valid = ~done
        row = (picked, (action == 5 + rock) & valid, out.reward * valid,
               diagnostic * (valid & active)[:, None], valid & active,
               jnp.all(state.rover == jnp.asarray(env.rocks)[rock], axis=-1) & valid,
               jnp.concatenate((state.rover[:4], action[:4, None], valid[:4, None]), axis=1))
        return (out.obs, out.state, cache, done | out.terminated | out.truncated,
                sampled | picked, key), row
    carry = (obs, state, cache, jnp.zeros(episodes, bool),
             jnp.zeros((episodes, env.k), bool), jax.random.key(seed + 1))
    _, rows = jax.lax.scan(advance, carry, jnp.arange(env.episode_length))
    picked, checks, rewards, diagnostic, active, visits, traces = rows
    return (picked.any(0), checks.sum(0), rewards.sum(0),
            diagnostic.sum((0, 1)), active.sum(), initial_good, visits.any(0),
            traces, jnp.max(jnp.where(active, diagnostic[:, :, 3], 0)),
            jnp.max(jnp.where(active, diagnostic[:, :, 2], 0)))


def summarize(values, rock):
    picked, checks, returns, diagnostic, count, initial_good, visits, _, endpoint_max, other_max = map(np.asarray, values)
    n = len(picked)
    p = picked.mean(0)
    z = 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    other = np.arange(picked.shape[1]) != rock
    good_mask = initial_good[:, other]
    other_picked = picked[:, other]
    selectivity = {}
    for quality, mask in (("good", good_mask), ("bad", ~good_mask)):
        selectivity[quality] = dict(picked=int((other_picked & mask).sum()),
                                  count=int(mask.sum()),
                                  probability=float((other_picked & mask).sum()/mask.sum()))
    return dict(episodes=n, sample_probability=p.tolist(),
                sample_ci95=np.stack((center-half, center+half), axis=1).tolist(),
                target_probability=float(p[rock]), checks=float(checks.mean()),
                mean_return=float(returns.mean()),
                decoded_before_after=(diagnostic[:2]/max(int(count), 1)).tolist(),
                decoded_nontarget_abs_change_mean=float(diagnostic[2]/max(int(count), 1)),
                decoded_endpoint_abs_error_mean=float(diagnostic[3]/max(int(count), 1)),
                decoded_endpoint_abs_error_max=float(endpoint_max),
                decoded_nontarget_abs_change_max=float(other_max),
                shift_norm_mean=float(diagnostic[4]/max(int(count), 1)),
                nontarget=selectivity, visit_probability=float(visits.mean()),
                checked_probability=float((checks > 0).mean()),
                visit_without_pickup_probability=float((visits & ~picked[:, rock]).mean()))


def evaluate(spec, env, params, fitted, episodes, seed, site, strength,
             *, intact_rows=None, trace_dir=None):
    result = []
    for rock in range(env.k):
        directions = {}
        for good, direction in ((1, "Good→Bad"), (0, "Bad→Good")):
            conditions = {}
            for condition in ("intact", "steered", "control"):
                if condition == "intact" and intact_rows is not None:
                    conditions[condition] = intact_rows[rock][direction]["intact"]
                    continue
                values = rollout(spec, env, params, fitted, episodes=episodes,
                    seed=seed, rock=rock, good=good, site=site,
                    strength=0. if condition == "intact" else strength,
                    control=condition == "control")
                conditions[condition] = summarize(values, rock)
                if trace_dir is not None:
                    traces = np.asarray(values[7])
                    np.savez_compressed(trace_dir / f"rock{rock}_good{good}_{condition}.npz", traces=traces)
                    examples = []
                    for episode in range(traces.shape[1]):
                        rows = traces[:, episode]
                        rows = rows[rows[:, 3].astype(bool)]
                        positions, actions = rows[:, :2], rows[:, 2]
                        at_rock = np.all(positions == np.asarray(env.rocks)[rock], axis=1)
                        route = [p.tolist() for t, p in enumerate(positions)
                                 if t == 0 or np.any(p != positions[t-1])]
                        examples.append(dict(episode=episode, route=route,
                            target_visited=bool(at_rock.any()),
                            target_sampled=bool(((actions == 4) & at_rock).any()),
                            check_counts=[int((actions == 5+i).sum()) for i in range(env.k)],
                            final_action=int(actions[-1])))
                    conditions[condition]["examples"] = examples
            directions[direction] = conditions
        result.append(directions)
    return result


def pooled(rows, direction, condition):
    cells = [r[direction][condition] for r in rows]
    result = dict(target=float(np.mean([c["target_probability"] for c in cells])),
        mean_return=float(np.mean([c["mean_return"] for c in cells])),
        decoded_before_after=np.mean([c["decoded_before_after"] for c in cells], axis=0).tolist(),
        endpoint_error_mean=float(np.mean([c["decoded_endpoint_abs_error_mean"] for c in cells])),
        endpoint_error_max=max(c["decoded_endpoint_abs_error_max"] for c in cells),
        nontarget_decoded_change_max=max(c["decoded_nontarget_abs_change_max"] for c in cells))
    for quality in ("good", "bad"):
        numerator = sum(c["nontarget"][quality]["picked"] for c in cells)
        denominator = sum(c["nontarget"][quality]["count"] for c in cells)
        result[f"nontarget_{quality}"] = numerator/denominator
    return result


def tune(spec, env, params, variants, episodes):
    evidence = {}
    intact = evaluate(spec, env, params, variants["encoder"], episodes, SEEDS["tune"], 2, 0.)
    for variant, fitted in variants.items():
        records = []
        for strength in ([2.] if variant == "encoder" else GRID):
            print(f"Tune {variant} λ={strength}", flush=True)
            rows = []
            for rock in range(env.k):
                directions = {}
                for good, direction in ((1, "Good→Bad"), (0, "Bad→Good")):
                    values = rollout(spec, env, params, fitted, episodes=episodes,
                        seed=SEEDS["tune"], rock=rock, good=good, site=2,
                        strength=strength, control=False)
                    directions[direction] = dict(steered=summarize(values, rock),
                                                intact=intact[rock][direction]["intact"])
                rows.append(directions)
            g, b = [pooled(rows, d, "steered") for d in ("Good→Bad", "Bad→Good")]
            drift = max(abs(pooled(rows, d, "steered")[f"nontarget_{q}"] -
                            pooled(rows, d, "intact")[f"nontarget_{q}"])
                        for d in ("Good→Bad", "Bad→Good") for q in ("good", "bad"))
            records.append(dict(strength=strength, good=g, bad=b,
                                selection_score=b["target"]-g["target"]-drift))
        eligible = [r for r in records if r["good"]["target"] <= .10 and r["bad"]["target"] >= .5]
        strength = (2. if variant == "encoder" else
                    min(r["strength"] for r in eligible) if eligible else 1.)
        selected = next(r for r in records if r["strength"] == strength)
        evidence[variant] = dict(strength=strength, selection_succeeded=bool(eligible),
                                selected_score=selected["selection_score"], sweep=records)
    return evidence, max(VARIANTS, key=lambda name: evidence[name]["selected_score"])


def write_report(output, report):
    best = report["best_variant"]
    lines = ["# RockSample r7 endpoint-calibrated steering (v2)", "", report["protocol"], "",
             f"Displayed variant: **{best}**, chosen using tuning only.", "",
             "Fit/test: 256/256 episodes per agent; tuning: 128; confirmation: 1,024 per",
             "rock/direction/condition/variant/site (smoke uses 8). Fresh confirmation seed 91423.",
             "Endpoints are measured at the intervened site, before downstream nonlinearities.",
             "Sites 0–2 = post-block pre-final-LN; site 3 = post-final-LN.", "",
             "## Tuning evidence", "",
             "| Variant | λ | Good→Bad pickup | Bad→Good pickup |",
             "|---|---:|---:|---:|"]
    for variant, tuning in report["tuning"].items():
        for row in tuning["sweep"]:
            lines.append(f"| {variant} | {row['strength']} | {row['good']['target']:.4f} | {row['bad']['target']:.4f} |")
        lines += [f"Selected {variant}: λ={tuning['strength']}; threshold met: "
                  f"{tuning['selection_succeeded']}; tuning score={tuning['selected_score']:.4f}.", ""]
    primary = report["agents"][0]
    lines += ["## Primary-agent variant comparison (site 2)", "",
              "| Variant | λ | G→B pickup | B→G pickup | G→B decoded after | B→G decoded after |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant, sites in primary["variants"].items():
        g, b = [pooled(sites["2"], d, "steered") for d in ("Good→Bad", "Bad→Good")]
        lines.append(f"| {variant} | {report['tuning'][variant]['strength']} | {g['target']:.4f} | "
                     f"{b['target']:.4f} | {g['decoded_before_after'][1]:.6f} | {b['decoded_before_after'][1]:.6f} |")
    lines += ["", "## Primary-agent selectivity, return and endpoint sanity (site 2)", "",
              "| Variant | Direction | Condition | Other Good | Other Bad | Return | Decoded before→after | Max endpoint error | Max other-coordinate change |",
              "|---|---|---|---:|---:|---:|---|---:|---:|"]
    for variant, sites in primary["variants"].items():
        for direction in ("Good→Bad", "Bad→Good"):
            for condition in ("intact", "steered", "control"):
                p = pooled(sites["2"], direction, condition)
                lines.append(f"| {variant} | {direction} | {condition} | {p['nontarget_good']:.4f} | "
                    f"{p['nontarget_bad']:.4f} | {p['mean_return']:.2f} | "
                    f"{p['decoded_before_after'][0]:.5f}→{p['decoded_before_after'][1]:.5f} | "
                    f"{p['endpoint_error_max']:.3g} | {p['nontarget_decoded_change_max']:.3g} |")
    for agent in report["agents"]:
        lines += ["", f"## Best variant: {agent['run_id']} seed {agent['seed']}", "",
                  "| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for i, row in enumerate(agent["variants"][best]["2"]):
            cells = [row[d][c] for d in ("Good→Bad", "Bad→Good") for c in ("intact", "steered", "control")]
            formatted = []
            for cell in cells:
                lo, hi = cell["sample_ci95"][i]
                formatted.append(f"{cell['target_probability']:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {i} | " + " | ".join(formatted) + " |")
        lines += ["", "| Variant | Site | G→B pickup | B→G pickup | R² by rock |",
                  "|---|---|---:|---:|---|"]
        for variant, sites in agent["variants"].items():
            for site, rows in sites.items():
                means = [pooled(rows, d, "steered")["target"] for d in ("Good→Bad", "Bad→Good")]
                lines.append(f"| {variant} | {site} | {means[0]:.3f} | {means[1]:.3f} | " +
                             ", ".join(f"{v:.3f}" for v in agent["r2"][int(site)]) + " |")
    lines += ["", "## Resistant-rock rollout evidence: primary agent, best variant, site 2", "",
              "Full confirmation rates below; examples are fixed episodes 0–3, not selected for outcomes.",
              "Raw position/action traces stay in ignored artifacts.", ""]
    for rock, row in enumerate(primary["variants"][best]["2"]):
        for direction in ("Good→Bad", "Bad→Good"):
            cell = row[direction]["steered"]
            resistant = cell["target_probability"] > .10 if direction == "Good→Bad" else cell["target_probability"] < .5
            if resistant:
                lines.append(f"- Rock {rock}, {direction}: pickup {cell['target_probability']:.3f}, "
                    f"visit {cell['visit_probability']:.3f}, checked {cell['checked_probability']:.3f}, "
                    f"target checks {cell['checks']:.2f}/episode; visited without pickup "
                    f"{cell['visit_without_pickup_probability']:.3f}.")
                for ex in cell.get("examples", [])[:2]:
                    lines.append(f"  Episode {ex['episode']}: route {ex['route']}; check counts "
                        f"by rock {ex['check_counts']}; target pickup {ex['target_sampled']}; "
                        f"final action {ex['final_action']}.")
    lines += ["", "Nontarget conditional rates pool the six nontarget rocks by initial quality.",
              "Wilson binomial 95% intervals do not represent training-seed uncertainty.",
              "Calibrated maps are solved in float64 and executed in float32; identity errors,",
              "condition numbers and row norms are recorded in results.json. Rank deficiency raises.",
              "Controls are decoder-null random directions with the counterfactual shift norm on",
              "their own current history, not a copied norm from a diverged steering trajectory.",
              "Intact episodes are reused across variants/sites because they are identical.",
              "Steering can still be off-manifold; endpoint calibration alone does not ensure selectivity."]
    (output / "findings.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rows = primary["variants"][best]["2"]
    for ax, direction in zip(axes[:2], ("Good→Bad", "Bad→Good")):
        for j, condition in enumerate(("intact", "steered", "control")):
            cells = [r[direction][condition] for r in rows]
            p = np.array([c["target_probability"] for c in cells])
            ci = np.array([c["sample_ci95"][i] for i, c in enumerate(cells)])
            ax.bar(np.arange(7)+(j-1)*.25, p, .25, label=condition,
                   yerr=np.maximum(0, np.stack((p-ci[:, 0], ci[:, 1]-p))), capsize=2)
        ax.set(title=direction, xlabel="Target rock (implementation index)", xticks=range(7), ylim=(0, 1.06))
    labels = ["G→B\nother Good", "G→B\nother Bad", "B→G\nother Good", "B→G\nother Bad"]
    for j, condition in enumerate(("intact", "steered", "control")):
        probabilities = [pooled(rows, d, condition)[f"nontarget_{q}"]
                         for d in ("Good→Bad", "Bad→Good") for q in ("good", "bad")]
        axes[2].bar(np.arange(4)+(j-1)*.25, probabilities, .25, label=condition)
    axes[2].set(title="Nontarget selectivity", xticks=range(4), xticklabels=labels, ylim=(0, 1.06))
    axes[0].set_ylabel("Pickup probability")
    axes[2].legend()
    fig.suptitle(f"{best}, λ={report['tuning'][best]['strength']}; primary agent, final block pre-LN")
    fig.tight_layout()
    fig.savefig(output / "pickup_probability.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(schema=2, smoke=args.smoke, seeds={**SEEDS, "confirm": CONFIRM_SEED}, grid=GRID,
        fit_episodes=8 if args.smoke else 256, test_episodes=8 if args.smoke else 256,
        tune_episodes=8 if args.smoke else 128, confirmation_episodes=8 if args.smoke else 1024,
        experiment_revision=bp.revision(bp.ROOT),
        harness_revision=bp.revision(bp.ROOT.parents[2] / "rl-harness"),
        jax_version=jax.__version__, sources={p.name: bp.sha256(p) for p in
            (Path(__file__), bp.ROOT / "model.py", bp.ROOT / "env.py", bp.ROOT / "belief_probe.py")},
        agents=[], protocol="Frozen r7 update-29 sampled policies, 7 current-Good marginals, fair-prior "
        "action/observation Bayesian targets; independent own-policy fit/test streams, all valid rows "
        "including reset, no warmup exclusion. Separate affine least squares (rcond 1e-6). "
        "Calibrated encoder = solve(EW,E); min-norm decoder = pseudoinverse(W). "
        "For each calibrated variant choose the smallest tuning λ with pooled Good→Bad pickup ≤0.10 "
        "AND Bad→Good pickup ≥0.5, else λ=1. Original encoder retains historical fallback λ=2. "
        "Choose displayed variant on tuning only by (Bad→Good pickup minus Good→Bad pickup minus "
        "maximum absolute nontarget conditional-rate drift from intact across directions/qualities); "
        "ties follow encoder, calibrated encoder, min-norm order. Freeze across agents/sites. "
        "Matched environment/action RNG across conditions; Bad→Good stops at first target pickup, "
        "Good→Bad continues every decision step. Cached K/V propagation preserved.")
    for run, seed in AGENTS[:1] if args.smoke else AGENTS:
        print(f"Loading {run} seed {seed}", flush=True)
        directory = bp.ROOT / "artifacts/belief_steering" / run
        directory.mkdir(parents=True, exist_ok=True)
        params, _, arm, provenance = bp.checkpoint(directory, seed, True,
            run_results=bp.ROOT / "ppo_bmax_r7/results" / run, update=29)
        env = rs.RockSampleParams.from_instance()
        spec = tm.TransformerSpec(env.obs_dim, env.num_actions, arm["d_model"],
                                 arm["n_layers"], arm["n_heads"], arm["context_len"])
        fitted, r2 = maps(spec, env, params, report["fit_episodes"])
        variants, calibration = calibrated_maps(fitted)
        if not report["agents"]:
            report["tuning"], report["best_variant"] = tune(spec, env, params, variants, report["tune_episodes"])
        agent = dict(run_id=run, seed=seed, checkpoint=provenance, r2=r2,
                     calibration=calibration, variants={})
        scratch = bp.ROOT / "artifacts" / args.output.name / f"{run}_seed{seed}"
        scratch.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(scratch / "maps.npz", decoder=np.asarray(fitted[0]),
            bias=np.asarray(fitted[1]), encoder=np.asarray(fitted[2]), null=np.asarray(fitted[3]))
        intact_rows = None
        for variant, fitted in variants.items():
            agent["variants"][variant] = {}
            strength = report["tuning"][variant]["strength"]
            for site in range(4):
                print(f"Confirm {run} seed {seed} {variant} site {site}, λ={strength}", flush=True)
                trace_dir = scratch / f"{variant}_site{site}"
                trace_dir.mkdir()
                rows = evaluate(spec, env, params, fitted, report["confirmation_episodes"], CONFIRM_SEED,
                    site, strength, intact_rows=intact_rows, trace_dir=trace_dir if site == 2 else None)
                agent["variants"][variant][str(site)] = rows
                if intact_rows is None:
                    intact_rows = rows
                (scratch / "progress.json").write_text(json.dumps(agent, indent=2, allow_nan=False)+"\n")
        report["agents"].append(agent)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_report(args.output, report)


if __name__ == "__main__":
    main()
