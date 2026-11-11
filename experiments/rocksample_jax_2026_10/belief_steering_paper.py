"""Paper-faithful plain-encoder marginal/joint RockSample steering (v5)."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_steering as bs

GRID = (0., .5, .75, .8, .85, .9, .95, 1.)
TUNE_SEED = 371839
CONFIRM_SEED = 471841
WARMUP = 8
REPRESENTATIONS = ("marginal", "joint")
DIRECTIONS = ("Good→Bad", "Bad→Good")
CONDITIONS = ("intact", "steered", "control", "random")
HISTORICAL = ("belief_steering_r7_20261110", "belief_steering_r7_20261110_v2",
              "belief_steering_r7_20261110_v3", "belief_steering_r7_20261110_v4_site0")


def full_joint(reduced):
    return jnp.concatenate((reduced, 1-jnp.sum(reduced, axis=-1, keepdims=True)), axis=-1)


def joint_target(decoded, rock, target):
    indices = jnp.arange(decoded.shape[-1])
    partner = indices ^ (1 << rock)
    return jnp.where(((indices >> rock) & 1) == target,
                     decoded + decoded[..., partner], 0.)


def retained_mask(valid):
    return valid & (np.arange(valid.shape[1])[None, :] >= WARMUP)


def fit_maps(spec, env, params, episodes, *, greedy=False):
    streams = []
    for seed in (bs.SEEDS["fit"], bs.SEEDS["test"]):
        rows = bs.bp.collect(spec, env, params, params, episodes=episodes, key_seed=seed, greedy=greedy)
        targets = bs.bp.targets(rows, env)
        mask = retained_mask(rows["valid"])
        if not mask.any():
            raise ValueError("no valid decisions remain after eight-step warmup")
        keep = mask[rows["valid"]]
        features = np.asarray(bs.replay(spec, params, rows["obs"]))[mask].astype(float)
        streams.append((features, dict(marginal=targets.marginals[keep], joint=targets.joint[keep])))
    (x, ys), (xt, yts) = streams
    fitted, scores, diagnostics = {}, {}, {}
    for representation in REPRESENTATIONS:
        rng = np.random.default_rng(bs.SEEDS["null"])
        full_y, full_yt = ys[representation], yts[representation]
        y = full_y[:, :-1] if representation == "joint" else full_y
        ds, offsets, es, nulls, score, details = [], [], [], [], [], []
        for site in range(spec.n_layers+1):
            d, b = bs.affine(x[:, site], y)
            e, _ = bs.affine(y, x[:, site])
            prediction = xt[:, site] @ d + b
            if representation == "joint":
                prediction = np.column_stack((prediction, 1-prediction.sum(-1)))
            residual = full_yt-prediction
            sst = ((full_yt-full_yt.mean(0))**2).sum(0)
            score.append(float(1-(residual**2).sum()/sst.sum()) if representation == "joint"
                         else [float(1-error/variance) if variance > 0 else None
                               for error, variance in zip((residual**2).sum(0), sst)])
            u, s, _ = np.linalg.svd(d, full_matrices=False)
            basis = u[:, s > s[0]*1e-6]
            random = rng.normal(size=(env.k, spec.d_model))
            random -= (random @ basis) @ basis.T
            norm = np.linalg.norm(random, axis=1, keepdims=True)
            if np.any(norm < 1e-10):
                raise ValueError("decoder has no numerically usable null direction")
            random /= norm
            details.append(dict(decoder_rank=basis.shape[1], null_dimension=spec.d_model-basis.shape[1],
                null_decoder_max_abs=float(np.max(np.abs(random.astype(np.float32) @ d.astype(np.float32)))),
                prediction_min=float(prediction.min()), prediction_max=float(prediction.max())))
            ds.append(d); offsets.append(b); es.append(e); nulls.append(random)
        fitted[representation] = tuple(jnp.asarray(v, dtype=jnp.float32) for v in (ds, offsets, es, nulls))
        scores[representation], diagnostics[representation] = score, details
    return fitted, scores, dict(fit_rows=len(x), test_rows=len(xt), maps=diagnostics)


def shift(h, d, b, e, null, rock, target, strength, control, active, representation):
    before = h @ d + b
    if representation == "joint":
        decoded = full_joint(before)
        desired = joint_target(decoded, rock, target)[:-1]
    else:
        desired = before.at[rock].set(target)
    delta = strength * ((desired-before) @ e) * active
    delta = jnp.where(control, jnp.linalg.norm(delta)*null[rock], delta)
    after = (h+delta) @ d + b
    if representation == "joint":
        bits = jnp.asarray(bs.bp.configuration_bits(7), dtype=jnp.float32)
        bm, am = full_joint(before) @ bits, full_joint(after) @ bits
    else:
        bm, am = before, after
    other = (am-bm).at[rock].set(0)
    diagnostic = jnp.stack((bm[rock], am[rock], jnp.max(jnp.abs(other)),
                            jnp.abs(am[rock]-target), jnp.linalg.norm(delta)))
    return h+delta, diagnostic


@partial(jax.jit, static_argnames=("spec", "env", "episodes", "site", "representation", "random_actions", "greedy"))
def rollout(spec, env, params, fitted, *, episodes, seed, rock, good, site,
            representation, strength, control, random_actions=False, greedy=False):
    d, b, e, null = fitted
    obs, state = jax.vmap(partial(bs.rs.reset, env))(
        jax.random.split(jax.random.key(seed), episodes))
    state = state._replace(qualities=state.qualities.at[:, rock].set(jnp.asarray(good, bool)))
    initial_good = state.qualities
    cache = jax.vmap(lambda _: bs.tm.empty_cache(spec))(jnp.arange(episodes))
    def advance(carry, t):
        obs, state, cache, done, sampled, key = carry
        active = ~sampled[:, rock] | jnp.asarray(good, bool)
        if random_actions:
            logits = jnp.zeros((episodes, env.num_actions))
            diagnostic = jnp.zeros((episodes, 5))
        else:
            def encode(c, o, enabled):
                diagnostics = []
                def transform(index, h):
                    if index != site:
                        return h
                    changed, diagnostic = shift(h.reshape(-1), d[index], b[index], e[index],
                        null[index], rock, 1-good, strength, control, enabled, representation)
                    diagnostics.append(diagnostic)
                    return changed.reshape(h.shape)
                z, c = bs.tm.encode_step(spec, params, c, o, residual_transform=transform)
                return z, c, diagnostics[0]
            z, cache, diagnostic = jax.vmap(encode)(cache, obs, active)
            logits, _ = bs.tm.heads(params, z)
        key, ak, ek = jax.random.split(key, 3)
        action = (jnp.argmax(logits, axis=-1).astype(jnp.int32) if greedy and not random_actions
                  else jax.random.categorical(ak, logits).astype(jnp.int32))
        out = jax.vmap(partial(bs.rs.step, env))(jax.random.split(ek, episodes), state, action)
        valid = ~done
        picked = ((action == 4)[:, None] & valid[:, None] &
                  jnp.all(state.rover[:, None] == jnp.asarray(env.rocks), axis=-1))
        row = (picked, (action == 5+rock) & valid, out.reward*valid,
               diagnostic*(valid & active)[:, None], valid & active,
               jnp.all(state.rover == jnp.asarray(env.rocks)[rock], axis=-1) & valid,
               jnp.concatenate((state.rover[:4], action[:4, None], valid[:4, None]), axis=1))
        return (out.obs, out.state, cache, done | out.terminated | out.truncated,
                sampled | picked, key), row
    carry = (obs, state, cache, jnp.zeros(episodes, bool), jnp.zeros((episodes, env.k), bool),
             jax.random.key(seed+1))
    _, rows = jax.lax.scan(advance, carry, jnp.arange(env.episode_length))
    picked, checks, rewards, diagnostic, active, visits, traces = rows
    return (picked.any(0), checks.sum(0), rewards.sum(0), diagnostic.sum((0, 1)), active.sum(),
            initial_good, visits.any(0), traces,
            jnp.max(jnp.where(active, diagnostic[:, :, 3], 0)),
            jnp.max(jnp.where(active, diagnostic[:, :, 2], 0)))


def pooled(rows, direction, condition):
    cells = [r[direction][condition] for r in rows]
    result = dict(target=float(np.mean([c["target_probability"] for c in cells])),
                  mean_return=float(np.mean([c["mean_return"] for c in cells])))
    for q in ("good", "bad"):
        result[f"nontarget_{q}"] = sum(c["nontarget"][q]["picked"] for c in cells)/sum(c["nontarget"][q]["count"] for c in cells)
    return result


def evaluate(spec, env, params, fitted, representation, episodes, seed, site, strengths,
             conditions=CONDITIONS, baselines=None, scratch=None, *, greedy=False):
    rows = []
    for rock in range(env.k):
        row = {}
        for good, direction in zip((1, 0), DIRECTIONS):
            cells = {}
            for condition in conditions:
                if baselines is not None and condition in ("intact", "random"):
                    cells[condition] = baselines[rock][direction][condition]
                    continue
                values = rollout(spec, env, params, fitted, episodes=episodes, seed=seed, rock=rock,
                    good=good, site=site, representation=representation,
                    strength=0. if condition in ("intact", "random") else strengths[direction],
                    control=condition == "control", random_actions=condition == "random", greedy=greedy)
                cell = bs.summarize(values, rock)
                if condition in ("intact", "random"):
                    cell = {k: v for k, v in cell.items() if not k.startswith(("decoded_", "shift_"))}
                cells[condition] = cell
                if scratch is not None:
                    np.savez_compressed(scratch/f"rock{rock}_good{good}_{condition}.npz",
                        picked=np.asarray(values[0]), checks=np.asarray(values[1]), returns=np.asarray(values[2]),
                        initial_good=np.asarray(values[5]), visits=np.asarray(values[6]), traces=np.asarray(values[7]))
            row[direction] = cells
        rows.append(row)
    return rows


def select_strength(records, direction, random_good):
    eligible = [r for r in records if (r["metrics"]["target"] <= random_good if direction == "Good→Bad"
                                      else r["metrics"]["target"] >= .5)]
    return dict(strength=min(r["strength"] for r in eligible) if eligible else 1.,
                selection_succeeded=bool(eligible), sweep=records)


def tune(spec, env, params, fitted, episodes, *, greedy=False):
    baseline = evaluate(spec, env, params, fitted["marginal"], "marginal", episodes, TUNE_SEED,
        2, dict.fromkeys(DIRECTIONS, 0.), conditions=("intact", "random"), greedy=greedy)
    random_good = pooled(baseline, "Good→Bad", "random")["target"]
    evidence = {}
    for representation in REPRESENTATIONS:
        records = {d: [] for d in DIRECTIONS}
        for strength in GRID:
            print(f"Tune {representation} λ={strength}", flush=True)
            rows = evaluate(spec, env, params, fitted[representation], representation, episodes,
                TUNE_SEED, 2, dict.fromkeys(DIRECTIONS, strength), conditions=("steered",), greedy=greedy) if strength else baseline
            for direction in DIRECTIONS:
                records[direction].append(dict(strength=strength,
                    metrics=pooled(rows, direction, "steered" if strength else "intact")))
        evidence[representation] = {d: select_strength(records[d], d, random_good) for d in DIRECTIONS}
    return dict(random_good_threshold=random_good, baselines=baseline, representations=evidence)


def write_report(output, report):
    lines = ["# RockSample r7 paper-protocol steering (v5)", "", report["protocol"], "",
             "## Tuning at the paper's site (site 2, after block 3 pre-LN)", "",
             f"Uniform-random Good pickup threshold: {report['tuning']['random_good_threshold']:.4f}.", "",
             "| Representation | Direction | λ | Target pickup | Other Good | Other Bad | Return |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for representation, directions in report["tuning"]["representations"].items():
        for direction, tuning in directions.items():
            for row in tuning["sweep"]:
                p = row["metrics"]
                lines.append(f"| {representation} | {direction} | {row['strength']} | {p['target']:.4f} | "
                             f"{p['nontarget_good']:.4f} | {p['nontarget_bad']:.4f} | {p['mean_return']:.2f} |")
    lines += ["", "| Representation | Direction | Selected λ | Criterion met |", "|---|---|---:|---|"]
    for representation, directions in report["tuning"]["representations"].items():
        for direction, tuning in directions.items():
            lines.append(f"| {representation} | {direction} | {tuning['strength']} | {tuning['selection_succeeded']} |")
    for agent in report["agents"]:
        lines += ["", f"## {agent['run_id']} seed {agent['seed']}", "",
                  "### Pooled confirmation: all sites and conditions", "",
                  "| Site | Representation | Direction | Condition | Target pickup | Other Good | Other Bad | Return |",
                  "|---|---|---|---|---:|---:|---:|---:|"]
        for site in range(4):
            for representation in REPRESENTATIONS:
                rows = agent["representations"][representation][str(site)]
                for direction in DIRECTIONS:
                    for condition in CONDITIONS:
                        p = pooled(rows, direction, condition)
                        lines.append(f"| {site} | {representation} | {direction} | {condition} | {p['target']:.4f} | "
                                     f"{p['nontarget_good']:.4f} | {p['nontarget_bad']:.4f} | {p['mean_return']:.2f} |")
        lines += ["", "### Held-out decoder R² (first eight decisions excluded)", "",
                  "| Site | Marginal R², rocks 0–6 | Joint overall R² |", "|---|---|---:|"]
        for site in range(4):
            lines.append(f"| {site} | " + ", ".join(f"{v:.3f}" for v in agent["r2"]["marginal"][site]) +
                         f" | {agent['r2']['joint'][site]:.4f} |")
        for site in (2, 0):
            for representation in REPRESENTATIONS:
                lines += ["", f"### Per-rock steered pickup: {representation}, site {site}", "",
                          "| Rock | Good→Bad [95% CI] | Bad→Good [95% CI] |", "|---|---:|---:|"]
                for rock, row in enumerate(agent["representations"][representation][str(site)]):
                    cells = []
                    for direction in DIRECTIONS:
                        c = row[direction]["steered"]
                        lo, hi = c["sample_ci95"][rock]
                        cells.append(f"{c['target_probability']:.3f} [{lo:.3f}, {hi:.3f}]")
                    lines.append(f"| {rock} | " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation and reporting limits", "",
              "Sites 0/1/2 are after blocks 1/2/3 before final LayerNorm; site 3 is after LayerNorm.",
              "Pickup intervals are Wilson binomial 95% intervals for each target rock, not training-seed uncertainty.",
              "Nontarget rates pool the other six rocks by their initial quality, not by the forced target quality.",
              "Joint overall R² is 1 minus summed squared error over all reconstructed 128 coordinates divided",
              "by summed held-out coordinate-wise variance. It is not the mean of coordinate-wise R² values.",
              "Decoded joint vectors and targets can be signed: no clipping, simplex projection or renormalization",
              "is added beyond reconstructing the omitted all-Good coordinate. Targets sum to one algebraically.",
              "Controls project random activation directions into each fitted decoder's numerical null space",
              "(relative SVD threshold 1e-6) and match the shift norm on their own current histories.",
              "Null dimensions and float32 orthogonality errors are recorded in results.json; the joint decoder",
              "may leave very few null dimensions. Endpoint calibration and nuisance regression are absent.",
              "Intact and random-action baselines are reused across representations/sites because their policies",
              "do not depend on the intervention. Random actions are uniform over all 12 actions, not a random",
              "trained policy. Evaluation includes reset and the first eight decisions. Raw arrays stay in artifacts."]
    (output/"findings.md").write_text("\n".join(lines)+"\n")


def figure(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    primary = report["agents"][0]
    for row, direction in enumerate(DIRECTIONS):
        for col, representation in enumerate(REPRESENTATIONS):
            ax = axes[row, col]
            for j, condition in enumerate(CONDITIONS):
                p = [pooled(primary["representations"][representation][str(site)], direction, condition)["target"]
                     for site in range(4)]
                ax.bar(np.arange(4)+(j-1.5)*.2, p, .2, label=condition)
            strength = report["tuning"]["representations"][representation][direction]["strength"]
            ax.set(title=f"{representation}: {direction}, λ={strength}", ylim=(0, 1.06),
                   xticks=range(4), xticklabels=["Block 1", "Block 2", "Block 3\n(paper site)", "Final LN"])
            if col == 0:
                ax.set_ylabel("Pooled target pickup probability")
    axes[0, 1].legend(ncol=2, fontsize=8)
    fig.suptitle("V5 paper-protocol steering: plain encoders; primary r7 agent")
    fig.tight_layout()
    fig.savefig(output/"pickup_by_site.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root = bs.bp.ROOT
    report = dict(schema=5, smoke=args.smoke, grid=GRID, warmup=WARMUP,
        fit_episodes=8 if args.smoke else 256, test_episodes=8 if args.smoke else 256,
        tune_episodes=8 if args.smoke else 128, confirmation_episodes=8 if args.smoke else 1024,
        seeds=dict(fit=bs.SEEDS["fit"], test=bs.SEEDS["test"], null=bs.SEEDS["null"],
                   tune=TUNE_SEED, confirm=CONFIRM_SEED), experiment_revision=bs.bp.revision(root),
        harness_revision=bs.bp.revision(root.parents[2]/"rl-harness"), jax_version=jax.__version__,
        sources={p.name: bs.bp.sha256(p) for p in
            (Path(__file__), Path(bs.__file__), root/"belief_probe.py", root/"env.py", root/"model.py")},
        historical_reports={name: bs.bp.sha256(root/"results"/name/"results.json") for name in HISTORICAL},
        agents=[], protocol="Paper-faithful Joint Reward port with RockSample adaptations: separate plain affine "
        "D (activation→belief) and E (belief→activation), no calibration and no nuisance regressors. "
        "Independent own-policy sampled-action fit/test streams, 256 episodes each; exclude decision indices "
        "0–7 from fitting and probe scoring only. Evaluation includes every decision. Marginal targets "
        "replace one of seven decoded Good probabilities. Joint D/E fit coordinates 0–126, omit all-Good "
        "configuration 127 and reconstruct it as 1 minus the sum. Set the target rock to a point mass "
        "while preserving the decoded joint over the other six rocks, then shift on 127 coordinates. "
        "Plain residual-times-encoder shifts at one site at a time, including KV-cache propagation. "
        "Tune on 128 episodes per rock/direction at site 2 using new seed 371839 and grid "
        "{0,0.5,0.75,0.8,0.85,0.9,0.95,1}. Independently select each representation/direction: "
        "Good→Bad smallest λ with pooled target pickup ≤ measured uniform-random Good pickup; "
        "Bad→Good smallest λ with pooled pickup ≥0.5. If no value qualifies, use λ=1. Freeze across "
        "all sites and agents. Confirm on seed 471841 with 1,024 episodes per rock/direction/condition. "
        "Matched environment/action RNG; stochastic actions; intact/steered/decoder-null control/uniform-random "
        "conditions. Bad→Good releases after first target pickup; Good→Bad continues every step. "
        "Smoke uses 8 episodes per stream and the primary agent only.")
    for run, seed in bs.AGENTS[:1] if args.smoke else bs.AGENTS:
        print(f"Loading {run} seed {seed}", flush=True)
        directory = root/"artifacts/belief_steering"/run
        directory.mkdir(parents=True, exist_ok=True)
        params, _, arm, provenance = bs.bp.checkpoint(directory, seed, True,
            run_results=root/"ppo_bmax_r7/results"/run, update=29)
        env = bs.rs.RockSampleParams.from_instance()
        spec = bs.tm.TransformerSpec(env.obs_dim, env.num_actions, arm["d_model"],
                                    arm["n_layers"], arm["n_heads"], arm["context_len"])
        fitted, scores, diagnostics = fit_maps(spec, env, params, report["fit_episodes"])
        if not report["agents"]:
            report["tuning"] = tune(spec, env, params, fitted, report["tune_episodes"])
        scratch = root/"artifacts"/args.output.name/f"{run}_seed{seed}"
        scratch.mkdir(parents=True, exist_ok=False)
        for representation, arrays in fitted.items():
            np.savez_compressed(scratch/f"{representation}_maps.npz", **dict(zip(("decoder", "bias", "encoder", "null"), map(np.asarray, arrays))))
        base_dir = scratch/"baselines"
        base_dir.mkdir()
        baselines = evaluate(spec, env, params, fitted["marginal"], "marginal", report["confirmation_episodes"],
            CONFIRM_SEED, 2, dict.fromkeys(DIRECTIONS, 0.), conditions=("intact", "random"), scratch=base_dir)
        agent = dict(run_id=run, seed=seed, checkpoint=provenance, r2=scores, diagnostics=diagnostics, representations={})
        for representation in REPRESENTATIONS:
            agent["representations"][representation] = {}
            strengths = {d: report["tuning"]["representations"][representation][d]["strength"] for d in DIRECTIONS}
            for site in range(4):
                print(f"Confirm {run} seed {seed}: {representation} site {site}, λ={strengths}", flush=True)
                trace_dir = scratch/f"{representation}_site{site}"
                trace_dir.mkdir()
                agent["representations"][representation][str(site)] = evaluate(spec, env, params,
                    fitted[representation], representation, report["confirmation_episodes"], CONFIRM_SEED,
                    site, strengths, baselines=baselines, scratch=trace_dir)
                (scratch/"progress.json").write_text(json.dumps(agent, indent=2, allow_nan=False)+"\n")
        report["agents"].append(agent)
        (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_report(args.output, report)
    figure(args.output, report)


if __name__ == "__main__":
    main()
