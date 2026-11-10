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
GRID = [0, .25, .5, .75, 1, 1.5, 2]


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


@partial(jax.jit, static_argnames=("spec", "env", "episodes"))
def rollout(spec, env, params, fitted, *, episodes, seed, rock, good,
            site, strength, control):
    d, b, e, null = fitted
    obs, state = jax.vmap(partial(rs.reset, env))(
        jax.random.split(jax.random.key(seed), episodes))
    state = state._replace(qualities=state.qualities.at[:, rock].set(jnp.asarray(good, bool)))
    cache = jax.vmap(lambda _: tm.empty_cache(spec))(jnp.arange(episodes))
    def advance(carry, t):
        obs, state, cache, done, sampled, key = carry
        active = ~sampled[:, rock] | jnp.asarray(good, bool)
        def encode(c, o, enabled):
            diagnostics = []
            def transform(index, h):
                changed, before, after = shift(h.reshape(-1), d[index], b[index],
                    e[index], null[index], rock, 1 - good, strength, control, enabled)
                diagnostics.append(jnp.stack((before, after)))
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
               diagnostic * (valid & active)[:, None], valid & active)
        return (out.obs, out.state, cache, done | out.terminated | out.truncated,
                sampled | picked, key), row
    carry = (obs, state, cache, jnp.zeros(episodes, bool),
             jnp.zeros((episodes, env.k), bool), jax.random.key(seed + 1))
    _, rows = jax.lax.scan(advance, carry, jnp.arange(env.episode_length))
    picked, checks, rewards, diagnostic, active = rows
    return (picked.any(0), checks.sum(0), rewards.sum(0),
            diagnostic.sum((0, 1)), active.sum())


def summarize(values, rock):
    picked, checks, returns, diagnostic, count = map(np.asarray, values)
    n = len(picked)
    p = picked.mean(0)
    z = 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(episodes=n, sample_probability=p.tolist(),
                sample_ci95=np.stack((center-half, center+half), axis=1).tolist(),
                target_probability=float(p[rock]), checks=float(checks.mean()),
                mean_return=float(returns.mean()),
                decoded_before_after=(diagnostic/max(int(count), 1)).tolist())


def evaluate(spec, env, params, fitted, episodes, seed, site, strength):
    result = []
    for rock in range(env.k):
        directions = {}
        for good, direction in ((1, "Good→Bad"), (0, "Bad→Good")):
            conditions = {}
            for condition in ("intact", "steered", "control"):
                values = rollout(spec, env, params, fitted, episodes=episodes,
                    seed=seed, rock=rock, good=good, site=site,
                    strength=0. if condition == "intact" else strength,
                    control=condition == "control")
                conditions[condition] = summarize(values, rock)
            directions[direction] = conditions
        result.append(directions)
    return result


def write_report(output, report):
    lines = ["# RockSample r7 marginal belief steering", "",
             report["protocol"], "", f"Selected λ = {report['strength']}; "
             f"selection criterion met: {report['selection_succeeded']}.", "",
             "| Tuning λ | Steered Good pickup | Intact Bad pickup |",
             "|---|---:|---:|"]
    lines.extend(f"| {r['strength']} | {r['good']:.4f} | {r['bad']:.4f} |"
                 for r in report["tuning"])
    lines.append("")
    for agent in report["agents"]:
        lines += [f"## {agent['run_id']} seed {agent['seed']}", "",
                  "| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for i, row in enumerate(agent["sites"]["2"]):
            cells = [row[direction][c] for direction in ("Good→Bad", "Bad→Good")
                     for c in ("intact", "steered", "control")]
            formatted = []
            for cell in cells:
                lo, hi = cell["sample_ci95"][i]
                formatted.append(f"{cell['target_probability']:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {i} | " + " | ".join(formatted) + " |")
        lines += ["", "| Site | Good→Bad pickup | Bad→Good pickup | Held-out per-rock R² |",
                  "|---|---:|---:|---|"]
        for site, rows in agent["sites"].items():
            means = [np.mean([r[direction]["steered"]["target_probability"] for r in rows])
                     for direction in ("Good→Bad", "Bad→Good")]
            lines.append(f"| {site} | {means[0]:.3f} | {means[1]:.3f} | " +
                         ", ".join(f"{v:.3f}" for v in agent["r2"][int(site)]) + " |")
        lines.append("")
    lines += ["Site indices 0–2 are post-block pre-final-LN; site 3 is post-final-LN.",
              "Intervals are marginal Wilson binomial 95% CIs, not training-seed uncertainty.",
              "Random controls use a fixed per-rock decoder-null direction with the norm of the",
              "counterfactual steering shift on their own current history (not a paired trajectory norm).",
              "Raw decoded beliefs are not clipped; separate least-squares encoders are not decoder inverses,",
              "so D(h′) need not equal the requested endpoint. Synthetic shifts can leave the policy manifold.",
              "All initial rock qualities are forced only for behavioral conditioning; decoder targets remain",
              "the ordinary fair-prior action/observation posterior. Bad→Good stops after first pickup;",
              "Good→Bad continues at every decision step."]
    (output / "findings.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, direction in zip(axes, ("Good→Bad", "Bad→Good")):
        for j, condition in enumerate(("intact", "steered", "control")):
            cells = [r[direction][condition] for r in report["agents"][0]["sites"]["2"]]
            p = np.array([c["target_probability"] for c in cells])
            ci = np.array([c["sample_ci95"][i] for i, c in enumerate(cells)])
            ax.bar(np.arange(7)+(j-1)*.25, p, .25, label=condition,
                   yerr=np.maximum(0, np.stack((p-ci[:, 0], ci[:, 1]-p))), capsize=2)
        ax.set(title=direction, xlabel="Target rock (implementation index)", xticks=range(7), ylim=(0, 1.06))
    axes[0].set_ylabel("Probability of sampling target rock")
    axes[1].legend()
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
    report = dict(schema=1, smoke=args.smoke, seeds=SEEDS, grid=GRID,
        fit_episodes=8 if args.smoke else 256, test_episodes=8 if args.smoke else 256,
        tune_episodes=8 if args.smoke else 128, confirmation_episodes=8 if args.smoke else 1024,
        experiment_revision=bp.revision(bp.ROOT),
        harness_revision=bp.revision(bp.ROOT.parents[2] / "rl-harness"),
        jax_version=jax.__version__, sources={p.name: bp.sha256(p) for p in
            (Path(__file__), bp.ROOT / "model.py", bp.ROOT / "env.py",
             bp.ROOT / "belief_probe.py")}, agents=[],
        protocol="Frozen r7 update-29 sampled policies, independent fit/test streams; "
        "all valid decision rows including reset priors, no warmup exclusion. "
        "Separate affine least-squares decoder and encoder per site (SVD rcond 1e-6). "
        "Tune the smallest grid λ whose pooled Good→Bad pickup is below intact Bad pickup; "
        "if none qualifies, use the grid endpoint and explicitly report selection failure. "
        "Freeze across agents/sites. Matched environment/action RNG across conditions. "
        "Bad→Good stops after the first target-rock sample; Good→Bad continues every step. "
        "K/V propagation is preserved.")
    episodes = 8 if args.smoke else 1024
    for run, seed in AGENTS[:1] if args.smoke else AGENTS:
        print(f"Loading {run} seed {seed}", flush=True)
        directory = bp.ROOT / "artifacts/belief_steering" / run
        directory.mkdir(parents=True, exist_ok=True)
        params, _, arm, provenance = bp.checkpoint(directory, seed, True,
            run_results=bp.ROOT / "ppo_bmax_r7/results" / run, update=29)
        env = rs.RockSampleParams.from_instance()
        spec = tm.TransformerSpec(env.obs_dim, env.num_actions, arm["d_model"],
                                 arm["n_layers"], arm["n_heads"], arm["context_len"])
        fitted, r2 = maps(spec, env, params, 8 if args.smoke else 256)
        if not report["agents"]:
            evidence = []
            for strength in GRID:
                print(f"Tune λ={strength}", flush=True)
                rates = []
                for rock in range(env.k):
                    rates.append([summarize(rollout(spec, env, params, fitted,
                        episodes=8 if args.smoke else 128, seed=SEEDS["tune"],
                        rock=rock, good=good, site=2, strength=lam, control=False),
                        rock)["target_probability"] for good, lam in ((1, strength), (0, 0.))])
                good, bad = np.mean(rates, axis=0)
                evidence.append(dict(strength=strength, good=float(good), bad=float(bad)))
            eligible = [r["strength"] for r in evidence if r["good"] < r["bad"]]
            report.update(strength=min(eligible) if eligible else GRID[-1],
                          selection_succeeded=bool(eligible), tuning=evidence)
        agent = dict(run_id=run, seed=seed, checkpoint=provenance, r2=r2, sites={})
        for site in range(4):
            print(f"Confirm site {site}, λ={report['strength']}", flush=True)
            agent["sites"][str(site)] = evaluate(spec, env, params, fitted, episodes,
                SEEDS["confirm"], site, report["strength"])
        report["agents"].append(agent)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_report(args.output, report)


if __name__ == "__main__":
    main()
