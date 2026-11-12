"""Markdown tables for the intention probes, steering onset, and per-rock steerability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.rocksample_jax_2026_10.belief_steering_cf import wilson
from experiments.rocksample_jax_2026_10.belief_steering_cf_report import (
    CHECKPOINTS, COORDINATES, DIRECTIONS)

SITES = ("block1", "block2", "block3", "final_norm")
FEATURES = dict(observables="Observables (position, step, previous action, symbol, sampled flags)",
                observables_belief="Observables + exact belief",
                position_x_belief="Position × (exact belief, sampled flags)",
                block1="Activations after block 1", block2="Activations after block 2",
                block3="Activations after block 3", final_norm="Activations after final norm")
MODES = dict(steer="Frozen steering (block 1)", current_token="Twin patch, block 1, new tokens",
             all_tokens="Twin patch, final activation")
PREDICTORS = dict(
    last_certainty="Belief certainty at last step before sampling/exit, |2m−1|",
    certainty="Mean belief certainty while unsampled",
    last_bad="Final exact belief when the rock is Bad",
    last_good="Final exact belief when the rock is Good",
    checks="Checks of the rock per episode",
    check_accuracy="Mean sensor accuracy of those checks",
    probe_r2_block1="Belief probe R², block 1",
    probe_r2_final="Belief probe R², final norm",
    reflect_r2="Affine fit R² of the rank-4 twin displacement, block 1",
    twin_norm="Twin displacement norm, block 1",
    twin_relative="Twin displacement norm / activation norm, block 1",
    top_share="Top singular direction's share of twin displacement, block 1")


def rate(successes, n):
    if not n:
        return "–"
    low, high = wilson(successes, n)
    return f"{successes / n:.3f} [{low:.3f}, {high:.3f}]"


def ranks(x):
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x))
    out[order] = np.arange(len(x))
    for value in np.unique(x):
        out[x == value] = out[x == value].mean()
    return out


def spearman(x, y):
    x, y = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pooled_outcome(reports, direction, kind, mode, key):
    cells = [r["cells"][f"{rock}|{direction}"][kind][mode][str(key)]
             for r in reports for rock in range(7)]
    total = lambda field: sum(c[field] for c in cells)
    weighted = lambda field, weight: (
        sum(c[field] * c[weight] for c in cells if c[field] is not None)
        / max(sum(c[weight] for c in cells if c[field] is not None), 1))
    return dict(eligible=total("eligible"), sampled=total("sampled"),
                intact=total("intact_sampled"), diverged=total("diverged"),
                diverge_delay=weighted("diverge_delay", "diverged"),
                at_onset=weighted("diverge_at_onset", "eligible"),
                sample_delay=weighted("sample_delay", "sampled"))


def steerability(intention, confirm):
    """One row per agent, checkpoint and rock: steering outcomes and candidate predictors."""
    rows = []
    for key, report in sorted(intention.items()):
        fit = confirm[key]["fit"]
        for rock in range(7):
            cells = confirm[key]["cells"]
            res = report["resolution"][rock]
            diag = fit["diagnostics"][rock]
            target = lambda d, name: cells[f"{rock}|{d}|{name}"]["target"]
            rows.append(dict(
                agent=key[0], update=key[1], rock=rock,
                bad_to_good=target("Bad→Good", "steer"), good_to_bad=target("Good→Bad", "steer"),
                interchange=target("Bad→Good", "interchange"),
                intact_good=target("Good→Bad", "intact"),
                last_certainty=res["last_certainty"], certainty=res["certainty"],
                last_bad=res["last_bad"], last_good=res["last_good"], checks=res["checks"],
                check_accuracy=res["check_accuracy"],
                probe_r2_block1=fit["r2"]["prob"][0][rock], probe_r2_final=fit["r2"]["prob"][3][rock],
                reflect_r2=diag["reflect_pca_all4_r2_site0"], twin_norm=diag["twin_norm_site0"],
                twin_relative=diag["twin_norm_site0"] / diag["h_norm_site0"],
                top_share=diag["spectrum_site0"][0]))
    return rows


def load(directory):
    out = {}
    for path in sorted(directory.glob("agent*_update*.json")):
        report = json.loads(path.read_text())
        out[(report["agent"], report["update"])] = report
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--confirm", type=Path, required=True,
                        help="directory with the held-out steering confirmation JSONs")
    args = parser.parse_args()
    intention, confirm = load(args.directory), load(args.confirm)
    lines = ["# Intention probes, steering onset and per-rock steerability", "",
             "Generated by `belief_steering_intention_report.py`. Rates pool seven target rocks",
             "and three agents, with Wilson 95% intervals over episodes.", ""]
    for update, label in CHECKPOINTS.items():
        reports = [r for (_, u), r in sorted(intention.items()) if u == update]
        if not reports:
            continue
        lines += [f"## {label} checkpoints (update {update}; agents "
                  f"{', '.join(str(r['agent']) for r in reports)})", "",
                  "### Next rock to be sampled: held-out linear readout", "",
                  "Eight classes (seven rocks and \"none\"). \"≥2 steps ahead\" drops the Sample",
                  "step itself and the step before it. Mean over agents (range).", "",
                  "| Features | Accuracy | Macro-F1 | Accuracy, ≥2 steps ahead | Macro-F1, ≥2 steps ahead |",
                  "|---|---|---|---|---|"]
        spread = lambda values: f"{np.mean(values):.3f} ({min(values):.3f}–{max(values):.3f})"
        for name, title in FEATURES.items():
            lines.append(f"| {title} | " + " | ".join(
                spread([r["probes"][name][subset][metric] for r in reports])
                for subset in ("all", "ahead") for metric in ("accuracy", "macro_f1")) + " |")
        majority = [max(r["label_share"]) for r in reports]
        lines += ["", f"Most frequent class share: {spread(majority)}.", "",
                  "### Next-rock readout under steering from the first step", "",
                  "Steps on which the target is still unsampled. \"Says target\": share of steps",
                  "on which the readout names the target rock; \"is target\": share on which the",
                  "target really is the next rock sampled in that rollout.", "",
                  "| Direction | Rollout | Is target | " + " | ".join(
                      f"Says target, {s}" for s in SITES) + " | " + " | ".join(
                      f"Accuracy, {s}" for s in SITES) + " |",
                  "|---|---|---|" + "---|" * (2 * len(SITES))]
        for d in DIRECTIONS:
            for kind, title in (("decode_intact", "Intact"), ("decode_steer", "Steered")):
                cells = [r["cells"][f"{rock}|{d}"][kind] for r in reports for rock in range(7)]
                steps = sum(c["steps"] for c in cells)
                mean = lambda f: sum(f(c) * c["steps"] for c in cells) / steps
                lines.append(
                    f"| {d} | {title} | {mean(lambda c: c['is_target']):.3f} | " + " | ".join(
                        f"{mean(lambda c: c[s]['says_target']):.3f}" for s in SITES) + " | "
                    + " | ".join(f"{mean(lambda c: c[s]['accuracy']):.3f}" for s in SITES) + " |")
        leads = sorted(int(k) for k in reports[0]["cells"]["0|Good→Bad"]["relative"]["steer"])
        lines += ["", "### Good rock: intervention starts `lead` steps before the intact Sample", "",
                  "Episodes in which the intact rollout samples the Good target. The steered",
                  "rollout shares every random draw, so it is identical to the intact one until",
                  "the onset. Lead 0 starts on the step whose action is the intact Sample.", "",
                  "| Intervention | " + " | ".join(f"lead {k}" for k in leads) + " |",
                  "|---|" + "---|" * len(leads)]
        for mode, title in MODES.items():
            out = [pooled_outcome(reports, "Good→Bad", "relative", mode, k) for k in leads]
            lines.append(f"| {title}: target sampled | " + " | ".join(
                rate(o["sampled"], o["eligible"]) for o in out) + " |")
            lines.append(f"| {title}: first changed action, steps after onset | " + " | ".join(
                f"{o['diverge_delay']:.2f}" for o in out) + " |")
            lines.append(f"| {title}: action changes on the onset step | " + " | ".join(
                f"{o['at_onset']:.3f}" for o in out) + " |")
        times = sorted(int(k) for k in reports[0]["cells"]["0|Good→Bad"]["fixed"]["steer"])
        for d in DIRECTIONS:
            lines += ["", f"### {d}: intervention starts at a fixed step", "",
                      "Episodes still running at the onset with the target unsampled; the share",
                      "in which the target is sampled at or after the onset.", "",
                      "| Rollout | " + " | ".join(f"step {t}" for t in times) + " |",
                      "|---|" + "---|" * len(times)]
            out = [pooled_outcome(reports, d, "fixed", "steer", t) for t in times]
            lines.append("| Intact | " + " | ".join(rate(o["intact"], o["eligible"]) for o in out) + " |")
            for mode, title in MODES.items():
                out = [pooled_outcome(reports, d, "fixed", mode, t) for t in times]
                lines.append(f"| {title} | " + " | ".join(
                    rate(o["sampled"], o["eligible"]) for o in out) + " |")
        cells = [r["cells"][f"{rock}|Good→Bad"]["timing"] for r in reports for rock in range(7)]
        lead = reports[0].get("trace_lead", 4)
        lines += ["", f"### Readout versus action after a mid-route onset (lead {lead})", "",
                  "Frozen steering starts before the intact Sample of a Good target. \"Readout",
                  "first\": the next-rock readout departs from the intact rollout's readout on an",
                  "earlier step than the action does; shares are over episodes where both change.", "",
                  "| Site | Says target before onset→Sample (intact) | Says target after onset (steered) "
                  "| Readout first | Same step | Action first | Readout delay | Action delay |",
                  "|---|---|---|---|---|---|---|---|"]
        acted = sum(c["acted"] for c in cells)
        action_delay = sum(c["action_delay"] * c["acted"] for c in cells) / acted
        for s in SITES:
            mean = lambda field, weight="eligible": (
                sum(c[s][field] * (c[weight] if weight in c else c[s][weight]) for c in cells)
                / sum(c[weight] if weight in c else c[s][weight] for c in cells))
            lines.append(
                f"| {s} | {mean('intact_says_target'):.3f} | {mean('steered_says_target'):.3f} | "
                f"{mean('plan_first', 'acted'):.3f} | {mean('same_step', 'acted'):.3f} | "
                f"{mean('action_first', 'acted'):.3f} | {mean('plan_delay', 'changed'):.2f} | "
                f"{action_delay:.2f} |")
        lines.append("")
    rows = steerability(intention, {k: v for k, v in confirm.items() if k in intention})
    lines += ["## Which rocks are harder to steer?", "",
              "Outcomes are the held-out confirmation's target sampling rates under frozen",
              "steering; predictors are measured on the intact policy's fit-seed rollouts.", "",
              "### Per rock, mean over three agents", ""]
    columns = ("bad_to_good", "interchange", "good_to_bad", "intact_good", "last_certainty",
               "last_bad", "checks", "check_accuracy", "reflect_r2", "twin_relative", "top_share",
               "probe_r2_block1")
    titles = ("B→G steered", "B→G rank-4 interchange", "G→B steered", "Intact Good sampled",
              "Final certainty", "Final belief, Bad", "Checks", "Check accuracy",
              "Affine fit R²", "Twin norm / activation norm", "Top direction share",
              "Probe R², block 1")
    for update, label in CHECKPOINTS.items():
        subset = [r for r in rows if r["update"] == update]
        if not subset:
            continue
        lines += [f"{label}:", "", "| Rock | " + " | ".join(titles) + " |",
                  "|---|" + "---|" * len(titles)]
        for rock in range(7):
            mine = [r for r in subset if r["rock"] == rock]
            lines.append(f"| {COORDINATES[rock]} | " + " | ".join(
                f"{np.mean([r[c] for r in mine]):.3f}" for c in columns) + " |")
        lines.append("")
    groups = sorted({(r["agent"], r["update"]) for r in rows})
    lines += ["### Rank correlation with Bad→Good steering", "",
              f"Spearman ρ over all {len(rows)} agent-checkpoint-rock rows, and the mean (range) of",
              f"the {len(groups)} within-agent-checkpoint correlations (seven rocks each). Descriptive",
              "only: seven rocks in one layout.", "",
              "| Predictor | ρ, all rows | ρ within agent-checkpoint, mean (range) | "
              "ρ with Good→Bad sampling, all rows |", "|---|---|---|---|"]
    for name, title in PREDICTORS.items():
        within = [spearman([r[name] for r in rows if (r["agent"], r["update"]) == g],
                           [r["bad_to_good"] for r in rows if (r["agent"], r["update"]) == g])
                  for g in groups]
        lines.append(
            f"| {title} | {spearman([r[name] for r in rows], [r['bad_to_good'] for r in rows]):+.2f} | "
            f"{np.nanmean(within):+.2f} ({np.nanmin(within):+.2f} to {np.nanmax(within):+.2f}) | "
            f"{spearman([r[name] for r in rows], [r['good_to_bad'] for r in rows]):+.2f} |")
    gap = [r["interchange"] - r["bad_to_good"] for r in rows]
    lines += ["", "Affine fit R² against the Bad→Good shortfall of steering relative to rank-4 twin "
              f"interchange: ρ = {spearman([r['reflect_r2'] for r in rows], gap):+.2f}.", ""]
    (args.directory / "tables.md").write_text("\n".join(lines))
    (args.directory / "steerability_rows.json").write_text(json.dumps(rows, allow_nan=False) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
