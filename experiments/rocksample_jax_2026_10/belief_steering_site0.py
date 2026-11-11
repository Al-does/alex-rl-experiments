"""Held-out replication of the site-0 choice made after v3 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_steering as bs

CONFIRM_SEED = 271829
SITE = 0
STRENGTH = 1.
HISTORICAL = ("belief_steering_r7_20261110", "belief_steering_r7_20261110_v2",
              "belief_steering_r7_20261110_v3")
DIRECTIONS = ("Good→Bad", "Bad→Good")
CONDITIONS = ("intact", "steered", "control")
PROTOCOL = (
    "Site 0 (after the first transformer block, before final LayerNorm) was selected "
    "after looking at v3 confirmation data. V4 is the held-out test of that choice, "
    "not a predeclared v3 site selection. Freeze the v3 nuisance-controlled encoder, "
    "λ=1, fit seed 41101, test seed 51203 and decoder-null seed 81517. Refit the same "
    "256-episode maps, with 256 independent decoder-test episodes. No tuning or "
    "selection on v4 data. Confirmation seed 271829 is new; 1,024 episodes per "
    "rock/direction/condition for each of the same three r7 update-29 agents. "
    "Matched environment/action RNG, sampled stochastic actions, cached K/V propagation "
    "and the matched-norm decoder-orthogonal random controls are unchanged. Bad→Good "
    "stops at first target pickup; Good→Bad continues every decision. Smoke uses 8 "
    "fit/test/confirmation episodes and the primary agent only."
)


def write_report(output, report):
    lines = ["# RockSample r7 site-0 held-out replication (v4)", "", report["protocol"], "",
             report["nuisance_features"], "",
             "Pickup entries are probability [Wilson binomial 95% CI]. Rock labels are",
             "implementation indices; intervals do not represent training-seed uncertainty.", ""]
    for agent in report["agents"]:
        rows = agent["rows"]
        lines += [f"## {agent['run_id']} seed {agent['seed']}", "",
                  "| Rock | Intact Good | Good→Bad | Control Good | Intact Bad | Bad→Good | Control Bad |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for rock, row in enumerate(rows):
            cells = []
            for direction in DIRECTIONS:
                for condition in CONDITIONS:
                    cell = row[direction][condition]
                    lo, hi = cell["sample_ci95"][rock]
                    cells.append(f"{cell['target_probability']:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {rock} | " + " | ".join(cells) + " |")
        lines += ["", "| Direction | Condition | Target pickup | Other Good | Other Bad | Mean return |",
                  "|---|---|---:|---:|---:|---:|"]
        for direction in DIRECTIONS:
            for condition in CONDITIONS:
                p = bs.pooled(rows, direction, condition)
                lines.append(f"| {direction} | {condition} | {p['target']:.4f} | "
                             f"{p['nontarget_good']:.4f} | {p['nontarget_bad']:.4f} | {p['mean_return']:.2f} |")
        errors = [bs.pooled(rows, d, "steered") for d in DIRECTIONS]
        lines += ["", "Held-out decoder R² by rock: " + ", ".join(f"{v:.3f}" for v in agent["r2"][SITE]) + ".",
                  f"Maximum steered endpoint error: {max(p['endpoint_error_max'] for p in errors):.3g}; "
                  f"maximum nontarget decoded-coordinate change: {max(p['nontarget_decoded_change_max'] for p in errors):.3g}.", ""]
    lines += ["## Illustrative paths", "",
              "Rock 0, episode 0 for both directions was fixed before v4 execution, not",
              "selected for a successful outcome. Intact/steered pairs use identical RNG",
              "and initial qualities. Paths show pre-action positions through termination",
              "or the 100-step horizon; stationary steps are collapsed. Sample markers",
              "include repeated pickups and empty-cell attempts; they are not quality labels.", "",
              "Nontarget pickup pools the six other rocks conditioned on initial quality.",
              "Controls match the counterfactual shift norm on their own current history,",
              "not the norm of a diverged steered trajectory. Endpoints are measured at",
              "the intervened site, before downstream nonlinearities. Raw arrays stay in artifacts."]
    (output / "findings.md").write_text("\n".join(lines) + "\n")


def figures(output, report, trace_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["agents"][0]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, direction in zip(axes[:2], DIRECTIONS):
        for j, condition in enumerate(CONDITIONS):
            cells = [r[direction][condition] for r in rows]
            p = np.array([c["target_probability"] for c in cells])
            ci = np.array([c["sample_ci95"][i] for i, c in enumerate(cells)])
            ax.bar(np.arange(7)+(j-1)*.25, p, .25, label=condition,
                   yerr=np.maximum(0, np.stack((p-ci[:, 0], ci[:, 1]-p))), capsize=2)
        ax.set(title=direction, xlabel="Target rock (implementation index)", xticks=range(7), ylim=(0, 1.06))
    for j, condition in enumerate(CONDITIONS):
        p = [bs.pooled(rows, d, condition)[f"nontarget_{q}"] for d in DIRECTIONS for q in ("good", "bad")]
        axes[2].bar(np.arange(4)+(j-1)*.25, p, .25, label=condition)
    axes[0].set_ylabel("Pickup probability")
    axes[2].set(title="Nontarget selectivity", xticks=range(4),
                xticklabels=["G→B\nother Good", "G→B\nother Bad", "B→G\nother Good", "B→G\nother Bad"], ylim=(0, 1.06))
    axes[2].legend()
    fig.suptitle("V4 held-out site-0 replication: nuisance-controlled encoder, λ=1; primary agent")
    fig.tight_layout()
    fig.savefig(output / "pickup_probability.png", dpi=180)
    plt.close(fig)

    env = bs.rs.RockSampleParams.from_instance()
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for row, (good, direction) in enumerate(zip((1, 0), DIRECTIONS)):
        for col, condition in enumerate(("intact", "steered")):
            ax = axes[row, col]
            with np.load(trace_dir / f"rock0_good{good}_{condition}.npz") as data:
                trace = data["traces"][:, 0]
            trace = trace[trace[:, 3].astype(bool)]
            positions = trace[:, :2]
            ax.plot(*positions.T, "o-", ms=3, alpha=.7)
            for rock, (x, y) in enumerate(env.rocks):
                ax.scatter(x, y, marker="D", s=100 if rock == 0 else 40,
                           color="red" if rock == 0 else "gray", zorder=3)
                ax.annotate(str(rock), (x+.1, y+.1))
            sampled = positions[trace[:, 2] == 4]
            if len(sampled):
                ax.scatter(*sampled.T, marker="x", color="black", s=70, label="sample action", zorder=4)
            ax.scatter(*positions[0], marker="s", color="green", s=70, label="start", zorder=4)
            ax.scatter(*positions[-1], marker="*", color="purple", s=90, label="last pre-action position", zorder=4)
            cell = rows[0][direction][condition]
            picked = cell["examples"][0]["target_sampled"]
            ax.set(title=f"{direction}: {condition}\nrock 0 picked: {picked}; {len(trace)} decisions",
                   xticks=range(5), yticks=range(5), xlim=(-.4, 4.4), ylim=(-.4, 4.4), aspect="equal")
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Fixed illustrative pairs: rock 0, episode 0, matched RNG; red diamond = target")
    fig.tight_layout()
    fig.savefig(output / "example_paths.png", dpi=180)
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
    previous = json.loads((root / "results" / HISTORICAL[-1] / "results.json").read_text())
    episodes = 8 if args.smoke else 1024
    report = dict(schema=4, smoke=args.smoke, protocol=PROTOCOL,
        nuisance_features=previous["nuisance_features"], site=SITE, strength=STRENGTH,
        variant="nuisance_controlled_encoder", fit_episodes=8 if args.smoke else 256,
        test_episodes=8 if args.smoke else 256, confirmation_episodes=episodes,
        seeds={k: bs.SEEDS[k] for k in ("fit", "test", "null")},
        experiment_revision=bs.bp.revision(root),
        harness_revision=bs.bp.revision(root.parents[2] / "rl-harness"), jax_version=jax.__version__,
        sources={p.name: bs.bp.sha256(p) for p in
                 (Path(__file__), Path(bs.__file__), root/"belief_probe.py", root/"model.py", root/"env.py")},
        historical_reports={name: bs.bp.sha256(root/"results"/name/"results.json") for name in HISTORICAL}, agents=[])
    report["seeds"]["confirm"] = CONFIRM_SEED
    primary_traces = None
    for run, seed in bs.AGENTS[:1] if args.smoke else bs.AGENTS:
        print(f"Loading {run} seed {seed}; frozen site={SITE}, λ={STRENGTH}, confirmation seed={CONFIRM_SEED}", flush=True)
        directory = root / "artifacts/belief_steering" / run
        directory.mkdir(parents=True, exist_ok=True)
        params, _, arm, provenance = bs.bp.checkpoint(directory, seed, True,
            run_results=root/"ppo_bmax_r7/results"/run, update=29)
        env = bs.rs.RockSampleParams.from_instance()
        spec = bs.tm.TransformerSpec(env.obs_dim, env.num_actions, arm["d_model"],
                                    arm["n_layers"], arm["n_heads"], arm["context_len"])
        fitted, r2, controlled, diagnostics = bs.maps(spec, env, params, report["fit_episodes"])
        variants, calibration = bs.calibrated_maps((fitted[0], fitted[1], controlled, fitted[3]))
        scratch = root / "artifacts" / args.output.name / f"{run}_seed{seed}"
        scratch.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(scratch/"maps.npz", decoder=np.asarray(fitted[0]), bias=np.asarray(fitted[1]),
                            nuisance_controlled_encoder=np.asarray(controlled), null=np.asarray(fitted[3]))
        rows = bs.evaluate(spec, env, params, variants["calibrated_encoder"], episodes,
                           CONFIRM_SEED, SITE, STRENGTH, trace_dir=scratch)
        if primary_traces is None:
            primary_traces = scratch
        report["agents"].append(dict(run_id=run, seed=seed, checkpoint=provenance, r2=r2,
                                     calibration=calibration, nuisance_design=diagnostics, rows=rows))
        (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_report(args.output, report)
    figures(args.output, report, primary_traces)


if __name__ == "__main__":
    main()
