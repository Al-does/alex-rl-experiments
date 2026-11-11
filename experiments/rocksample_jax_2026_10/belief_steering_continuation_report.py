"""Render the matched update-29/update-316 steering comparison."""

import argparse
from pathlib import Path

from experiments.rocksample_jax_2026_10 import belief_steering_continuation as cont

PROTOCOLS = (("v4", "nuisance-controlled calibrated"), ("v5", "marginal"), ("v5", "joint"))
DIRECTIONS = cont.paper.DIRECTIONS


def rows(state, protocol, representation, site):
    data = state[protocol]
    return data["sites"][str(site)] if protocol == "v4" else data["representations"][representation][str(site)]


def load(output):
    manifest = cont.read(output/"manifest.json")
    if not manifest["mappings"]:
        raise ValueError("no agents in comparison")
    agents = []
    for mapping in manifest["mappings"]:
        pair = {}
        for update in (29, 316):
            state = cont.read(output/f"{mapping['source_run']}_seed{mapping['seed']}_update{update}.json")
            if state["mapping"] != mapping or state["checkpoint"]["update"] != update:
                raise ValueError("comparison record mapping/update mismatch")
            for protocol, representation in PROTOCOLS:
                for site in range(4):
                    cells = rows(state, protocol, representation, site)
                    if len(cells) != 7:
                        raise ValueError("incomplete rock table")
                    conditions = {"intact", "steered", "control"} | ({"random"} if protocol == "v5" else set())
                    for rock in cells:
                        for direction in DIRECTIONS:
                            if set(rock[direction]) != conditions or any(
                                    c["episodes"] != manifest["confirmation_episodes"] for c in rock[direction].values()):
                                raise ValueError("incomplete condition or episode budget")
            pair[update] = state
        agents.append(pair)
    return manifest, agents


def pooled(pair, update, protocol, representation, site, direction, condition="steered"):
    return cont.paper.pooled(rows(pair[update], protocol, representation, site), direction, condition)


def label(pair):
    m = pair[29]["mapping"]
    return f"{m['source_run']} seed {m['seed']} → {m['continuation_run']}"


def report(output, manifest, agents):
    lines = ["# RockSample r7: matched steering at updates 29 and 316", "",
        "Update 29: 30,408,704 lifetime steps. Update 316: 331,350,016 lifetime steps.",
        "Continuation mappings are discovered from continuation_source.json and checked against the",
        "historical checkpoint SHA-256, size and URI, not inferred from launch order. Each final",
        "checkpoint is size/SHA-256 verified against its canonical B2 durability manifest before",
        "unpickling; recipe architecture and every parameter shape are checked (d128, three blocks,",
        "four heads, context 32). This is offline CPU analysis without retraining.", "",
        "## Frozen protocols and matched comparison", "",
        "V4 protocol: nuisance-controlled calibrated encoder, λ=1 at all four sites, no tuning.",
        "Fit/test/null seeds 41101/51203/81517; confirmation seed 271829. No warmup exclusion.",
        "V5 protocol: plain affine encoder/decoder, marginal and reduced 127-coordinate joint targets;",
        "fit/test/null seeds as above; exclude decisions 0–7 from fitting/scoring only. Every evaluation",
        "decision is included. Tune on seed 371839 at site 2, grid {0,.5,.75,.8,.85,.9,.95,1}; select",
        "each representation/direction independently. Good→Bad: smallest λ with target pickup ≤",
        "uniform-random Good pickup on that tuning seed; Bad→Good: smallest λ with pickup ≥.5;",
        "otherwise λ=1. Confirmation seed 471841. Both protocols release Bad→Good after first pickup,",
        "use matched environment/action RNG and matched-norm decoder-null controls with cached K/V propagation.",
        "Budgets are 256 fit / 256 test / 128 tuning / 1,024 confirmation episodes per rock/direction/condition.", "",
        "As requested here, V5 is re-tuned separately for each checkpoint, including each update-29 agent.",
        "Historical v5 tuned only the primary agent and froze its strengths across agents. Consequently",
        "a re-tuned update-29 row is not necessarily identical to that historical row. Historical confirmation",
        "cells are reused only if both direction strengths match exactly; v4 site-0 historical cells are",
        "also reused. New update-29 v4 sites use the v4 confirmation seed, not v3's different seed.",
        "All v1–v5 directories remain untouched. Comparisons use fitted maps from each checkpoint's own",
        "sampled-policy streams on matched seeds; histories and fit-row counts can differ between checkpoints.", "",
        "Sites 0/1/2 are after blocks 1/2/3 pre-final-LN; site 3 is after final LayerNorm.",
        "Tables use implementation rock indices. Other Good/Bad conditions the six nontarget rocks on",
        "their initial quality. Wilson 95% intervals reflect episodes, not training-seed uncertainty.", "",
        "## Verified continuation checkpoints", "",
        "| Source agent | Continuation run | Bytes | SHA-256 |", "|---|---|---:|---|"]
    for pair in agents:
        m, p = pair[29]["mapping"], pair[316]["checkpoint"]
        lines.append(f"| {m['source_run']} seed {m['seed']} | {m['continuation_run']} | {p['size_bytes']} | {p['sha256']} |")
    lines += ["", "## Selected V5 strengths (re-tuned per checkpoint)", "",
              "| Agent | Update | Representation | Direction | λ | Criterion met | Random Good threshold |",
              "|---|---:|---|---|---:|---|---:|"]
    for pair in agents:
        for update in (29, 316):
            tuning = pair[update]["v5"]["tuning"]
            for representation in cont.paper.REPRESENTATIONS:
                for direction in DIRECTIONS:
                    t = tuning["representations"][representation][direction]
                    lines.append(f"| {label(pair)} | {update} | {representation} | {direction} | {t['strength']} | "
                                 f"{t['selection_succeeded']} | {tuning['random_good_threshold']:.4f} |")
    for pair in agents:
        lines += ["", f"## {label(pair)}", ""]
        for protocol, representation in PROTOCOLS:
            lines += [f"### {protocol}: {representation}, pooled comparison", "",
                      "Each metric is update 29 → update 316.", "",
                      "| Site | Direction | Condition | Target pickup | Other Good | Other Bad | Return |",
                      "|---:|---|---|---:|---:|---:|---:|"]
            for site in range(4):
                for direction in DIRECTIONS:
                    for condition in ("intact", "steered", "control") + (("random",) if protocol == "v5" else ()):
                        values = [pooled(pair, u, protocol, representation, site, direction, condition) for u in (29, 316)]
                        cells = [f"{values[0][k]:.4f} → {values[1][k]:.4f}" for k in
                                 ("target", "nontarget_good", "nontarget_bad", "mean_return")]
                        lines.append(f"| {site} | {direction} | {condition} | " + " | ".join(cells) + " |")
            for site in (0, 2):
                lines += ["", f"#### Per-rock steered pickup, site {site}", "",
                          "| Rock | Update 29 Good→Bad | Update 316 Good→Bad | Update 29 Bad→Good | Update 316 Bad→Good |",
                          "|---:|---:|---:|---:|---:|"]
                for rock in range(7):
                    cells = []
                    for direction in DIRECTIONS:
                        for update in (29, 316):
                            c = rows(pair[update], protocol, representation, site)[rock][direction]["steered"]
                            lo, hi = c["sample_ci95"][rock]
                            cells.append(f"{c['target_probability']:.3f} [{lo:.3f}, {hi:.3f}]")
                    lines.append(f"| {rock} | " + " | ".join(cells) + " |")
        lines += ["", "### Held-out probe R²", "",
                  "V4 uses all valid decisions; V5 excludes the first eight. Joint R² aggregates squared",
                  "errors and coordinate-wise held-out variance across all 128 reconstructed coordinates.", "",
                  "| Update | Site | V4 marginal R², rocks 0–6 | V5 marginal R², rocks 0–6 | V5 joint overall R² |",
                  "|---:|---:|---|---|---:|"]
        for update in (29, 316):
            for site in range(4):
                v4 = ", ".join(f"{v:.3f}" for v in pair[update]["v4"]["r2"][site])
                v5 = pair[update]["v5"]["r2"]
                marginal = ", ".join(f"{v:.3f}" for v in v5["marginal"][site])
                lines.append(f"| {update} | {site} | {v4} | {marginal} | {v5['joint'][site]:.4f} |")
        lines += ["", "### Full V5 tuning grids", "",
                  "| Update | Representation | Direction | λ | Target pickup | Other Good | Other Bad | Return |",
                  "|---:|---|---|---:|---:|---:|---:|---:|"]
        for update in (29, 316):
            for representation, directions in pair[update]["v5"]["tuning"]["representations"].items():
                for direction, tuning in directions.items():
                    for t in tuning["sweep"]:
                        p = t["metrics"]
                        lines.append(f"| {update} | {representation} | {direction} | {t['strength']} | {p['target']:.4f} | "
                            f"{p['nontarget_good']:.4f} | {p['nontarget_bad']:.4f} | {p['mean_return']:.2f} |")
    lines += ["", "## Interpretation limits", "",
        "V4 holds λ fixed, while V5 permits independent re-tuning per checkpoint. If selected strengths",
        "differ, V5 is not a fixed-strength comparison. Neither protocol selects a site on",
        "these confirmation data. A lower Good→Bad target rate or higher Bad→Good rate is not sufficient",
        "for improved selective steering: inspect nontarget pickup, intact behavior, controls and return.",
        "Calibration endpoints are measured immediately at the intervened site, not after downstream nonlinearities.",
        "Joint decoded vectors may be signed; no clipping or simplex projection is added. Controls use",
        "the existing relative-SVD-threshold 1e-6 numerical decoder null space and match norms on their",
        "own current histories. V5 null dimensions/orthogonality diagnostics and fitting row counts are",
        "recorded in each checkpoint JSON. Decoder R² is not evidence of causal belief use."]
    (output/"findings.md").write_text("\n".join(lines)+"\n")


def figure(output, agents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    for column, (protocol, representation) in enumerate(PROTOCOLS):
        for row, direction in enumerate(DIRECTIONS):
            ax = axes[row, column]
            for update, color in ((29, "tab:blue"), (316, "tab:orange")):
                for condition, style, marker in (("steered", "-", "o"), ("intact", ":", None), ("control", "--", "x")):
                    values = [pooled(agents[0], update, protocol, representation, site, direction, condition)["target"]
                              for site in range(4)]
                    ax.plot(range(4), values, linestyle=style, marker=marker, color=color,
                            label=f"update {update}: {condition}")
            ax.set(title=f"{protocol}: {representation}\n{direction}", ylim=(-.02, 1.04),
                   xticks=range(4), xticklabels=["Block 1", "Block 2", "Block 3\n(paper site)", "Final LN"])
            if column == 0:
                ax.set_ylabel("Pooled target pickup probability")
    axes[0, 2].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("RockSample r7 primary agent: update 29 vs update 316")
    fig.tight_layout()
    fig.savefig(output/"pickup_by_site.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    manifest, agents = load(output)
    report(output, manifest, agents)
    figure(output, agents)
    cont.write(output/"report_source.json", dict(source=Path(__file__).name, sha256=cont.bs.bp.sha256(Path(__file__))))


if __name__ == "__main__":
    main()
