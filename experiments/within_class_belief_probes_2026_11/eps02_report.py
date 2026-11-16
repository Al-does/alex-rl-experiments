"""Render the eps02 seed-aggregated reviewer-response tables and vector figure."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from .eps02 import OUTPUT, ROOT, STUDIES
from .eps02_metrics import values

START = "<!-- eps02-within-class:start -->"
END = "<!-- eps02-within-class:end -->"


def number(value):
    return "NA" if value is None else f"{value:.4f}"


def interval(metric):
    value = number(metric["mean"])
    ci = metric["ci95"]
    return value if ci is None else f"{value} [{ci[0]:.4f}, {ci[1]:.4f}]"


def append_section(path, section):
    text = path.read_text()
    block = START+"\n"+section.rstrip()+"\n"+END
    if START in text:
        if END not in text:
            raise ValueError("incomplete eps02 report block")
        left, rest = text.split(START, 1)
        _, right = rest.split(END, 1)
        text = left+block+right
    else:
        text = text.rstrip()+"\n\n"+block+"\n"
    path.write_text(text)


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "|"+"|".join("---" for _ in headers)+"|",
                      *["| "+" | ".join(map(str, row))+" |" for row in rows]])


def markdown(report):
    sections = ["## PR #234 eps02: affine belief accessibility inside greedy-action classes",
        "CPU-only frozen-checkpoint analysis. Source: `"+report["provenance"]["source_revision"]+"` from fetched main, imported with `git archive`; no merge/rebase. Reward-factor studies were excluded.",
        "### Inventory and target",
        "**48 runs / 4,896 durable checkpoints:** three transformer conditions, seeds 42–56 at 2,012,217,344 steps; three seed-42 continuations at 4,024,434,688 steps. The original seed-42 runs are the campaign's seed 42, not additional replicates. No other eps02 Ladder architectures have published runs in this main snapshot.",
        "Target: full **three-state decision-time filtered belief**, recomputed from tokens and executed actions using emissions `[[.1125,.0375,.85],[.0375,.1125,.85],[.01,.01,.98]]`; never reward-conditioned. Delta is b1−b2, rho is b3. Transitions, uniform prior, effect size 2.5, temperature 1.5, entropy 0 and no epsilon-greedy are unchanged from v2.",
        "Complete canonical-manifest inventory (B2 prefixes exclude the bucket; bucket is `"+report["inventory"][0]["b2_bucket"]+"`). Counts are read from B2, cross-checked against archived checkpoint reports, not inferred from intended budgets:"]
    sections.append(table(["Study", "Leaf", "Seed", "Steps", "B2 prefix", "Saved checkpoints"],
        [[r["study"], r["leaf"], r["seed"], r["steps"], "`"+r["b2_prefix"]+"`", r["saved_checkpoints"]]
         for r in report["inventory"]]))
    sections += ["### Archive reproduction — annotated rows only",
        "Each checkpoint is byte-count/SHA-256 verified against its canonical B2 durability manifest. The original global affine probe is reproduced on CPU; discrepancies are annotations only at max(.01, 5 × whole-test-episode-bootstrap SE). Both R² and 1−R² comparisons are retained in JSON."]
    annotations = []
    checks = 0
    for run in report["runs"]:
        for stage in ("init", "final", "continuation_join"):
            if stage not in run:
                continue
            check = run[stage]["archive_check"]
            checks += 1
            if check["annotation"]:
                annotations.append([run["arm"], run["seed"], "4B" if run["study"] == STUDIES[1] else "2B",
                    stage, number(check["archived"]), number(check["reproduced"]),
                    number(check["delta"]), number(check["annotation_threshold"])])
    sections.append(f"{checks} archive comparisons ({len(annotations)} annotated); continuation initialization reuses the original parent's zero-step initialization. Repeated init comparisons are not independent replicates.")
    sections.append(table(["Arm", "Seed", "Run", "Stage", "Archive R²", "CPU R²", "Δ", "Threshold"], annotations)
                    if annotations else "No annotated rows.")
    sections += ["### Primary: equal-weight training-seed means and 95% seed-bootstrap CIs",
        "Each condition has 15 independent training seeds. Intervals use **10,000 whole-seed resamples**, with the trained-minus-matched contrast paired within seed. All within-class claims, scalar/head probes and controls below exclude one-class finals; global R² and between-class fractions use all 15. Undefined metrics retain their own coverage lists in JSON. One-class within scores equal global accessibility, not additional conditional evidence.",
        "Between-class fraction = 1 − within-class test SST/global test SST, the descriptive class-mean oracle's global R². A genuinely held-out class-only baseline, using fit-class means, is also saved; it need not equal this fraction exactly."]
    main, probes, own, descriptive, descriptive_probes = [], [], [], [], []
    for group in report["aggregates"].values():
        all_seeds, eligible = group["all_seed_descriptive"], group["within_class_eligible"]
        a, e = all_seeds["metrics"], eligible["metrics"]
        main.append([group["condition"], f"{eligible['n_seeds']}/15", interval(a["global_trained"]),
            interval(a["global_init"]), interval(a["between_class_fraction"]), interval(e["within_trained"]),
            interval(e["within_matched_init"]), interval(e["final_minus_matched_init"]),
            f"{eligible['positive_final_minus_matched_init']}/{eligible['n_defined_differences']}",
            len(all_seeds["one_class_seeds"])])
        probes.append([group["condition"], *[interval(e[k]) for k in
            ("head_blind", "logits", "delta", "rho", "history_shuffle", "random_input", "label_permutation")]])
        own.append([group["condition"], interval(e["within_own_init"]), interval(e["global_matched_init"]),
            interval(e["history_shuffle_matched_init"]), interval(e["random_input_matched_init"]),
            interval(e["label_permutation_matched_init"])])
        descriptive.append([group["condition"], interval(a["within_trained"]), interval(a["within_matched_init"]),
            interval(a["final_minus_matched_init"]),
            f"{all_seeds['positive_final_minus_matched_init']}/{all_seeds['n_defined_differences']}",
            interval(a["history_shuffle"]), interval(a["random_input"]), interval(a["label_permutation"])])
        descriptive_probes.append([group["condition"], interval(a["global_matched_init"]),
            interval(a["within_own_init"]), *[interval(a[k]) for k in ("head_blind", "logits", "delta", "rho")]])
    sections.append(table(["Condition", "Eligible seeds", "Global trained", "Global own init", "Between-class fraction",
        "Within trained", "Within matched init", "Final − matched init", "Positive seeds", "One-class seeds"], main))
    sections.append(table(["Condition", "Head-blind", "Logits", "Delta", "Rho", "Shuffle", "Random input", "Label permutation"], probes))
    sections += ["### Initialization and controls",
        "Own-policy initialization is secondary: it changes visitation/classes as well as parameters. Matched initialization keeps the final policy's exact complete observation/previous-action streams, beliefs, classes and row ordering, replaying the initialized encoder from an empty cache per episode. Controls are separately refitted using original targets/classes; one fixed draw each, not permutation significance tests.",
        table(["Condition", "Within own init (final-eligible subset)", "Global matched init", "Matched shuffle", "Matched random", "Matched labels"], own),
        "### All-seed descriptive values — includes one-class/global-accessibility cases",
        "This table reports excluded one-class runs without treating them as within-class evidence. The main conditional claims and CIs above use only eligible finals.",
        table(["Condition", "Trained", "Matched init", "Difference", "Positive seeds", "Shuffle", "Random", "Labels"], descriptive),
        table(["Condition", "Global matched init", "Own-init within (descriptive)", "Head-blind", "Logits", "Delta", "Rho"], descriptive_probes),
        "### Secondary: seed 42, 2B parent and 4B continuation versus original initialization",
        "These are individual-seed point estimates, without training-seed CIs. Both 2B and 4B use the parent's original zero-step initialization. Each matched-init probe uses that row's own final-policy histories. Own-history differences between 2B and 4B also change visitation and class proportions; they are not isolated representational effects. The continuation restarted Adam moments at the join."]
    secondary = []
    for run in report["runs"]:
        if run["seed"] != 42:
            continue
        v = values(run)
        secondary.append([run["condition"], "4B" if run["study"] == STUDIES[1] else "2B",
            run["final"]["step"], number(v["global_trained"]), number(v["global_init"]),
            number(v["between_class_fraction"]), number(v["within_trained"]), number(v["within_matched_init"]),
            number(v["final_minus_matched_init"]), number(v["within_own_init"]),
            number(v["head_blind"]), number(v["logits"]), number(v["delta"]), number(v["rho"]),
            str(run["final"]["one_class"])])
    sections.append(table(["Condition", "Stage", "Steps", "Global", "Global init", "Between fraction", "Within trained",
        "Matched init", "Difference", "Own init", "Head-blind", "Logits", "Delta", "Rho", "One class"], secondary))
    sections += ["### Per-seed final and matched-init results",
        "Per-class R², SSE/SST, excluded-row counts and full fit/test class counts for all probes and controls are retained in JSON; a missing class is NA, not zero."]
    seeds = []
    for run in report["runs"]:
        v = values(run)
        seeds.append([run["condition"], run["seed"], "4B" if run["study"] == STUDIES[1] else "2B",
            number(v["global_trained"]), number(v["between_class_fraction"]), number(v["within_trained"]),
            number(v["within_matched_init"]), number(v["final_minus_matched_init"]),
            "/".join(map(str, run["final"]["test_counts"])), str(run["final"]["one_class"])])
    sections.append(table(["Condition", "Seed", "Stage", "Global", "Between fraction", "Within trained", "Matched init",
                           "Difference", "Test classes 0/1/2", "One class"], seeds))
    sections += ["### Interpretation and limitations",
        "High class-centred held-out R² above the controls in eligible multi-class policies supports **continuous affine belief accessibility inside their own greedy-action regions**, beyond a class-only predictor. The paired final-minus-matched-init contrast asks whether training increases accessibility on the exact same histories/classes. Neither finding establishes that the policy causally uses that information or implements Bayesian computation.",
        "One-class runs are excluded from conditional claims. Seed intervals condition on fixed replay/probe streams, fit data and one perturbation draw per control; they do not include every source of Monte Carlo or probe-fitting uncertainty. Class-specific fits require at least two fit rows; constant or empty targets have undefined R². Delta and rho are task-derived coordinates, not universally action-irrelevant nuisance variables: both can affect action choice in Full-State. Head-blind removes own-head logit differences, not nonlinear or causal action information.",
        "![eps02 seed-aggregated probes](ladder_eps02_within_class.png)",
        "Vector figure: [PDF](ladder_eps02_within_class.pdf). JSON: [full results and provenance](ladder_eps02_within_class.json)."]
    findings = []
    for group in report["aggregates"].values():
        eligible = group["within_class_eligible"]
        if not eligible["n_seeds"]:
            findings.append(group["condition"]+": all finals are one-class, so this condition supplies no additional action-conditioned evidence.")
            continue
        metric = eligible["metrics"]["final_minus_matched_init"]
        ci = metric["ci95"]
        direction = ("increases" if ci and ci[0] > 0 else
                     "decreases" if ci and ci[1] < 0 else "shows no clear mean change in")
        findings.append(f"{group['condition']}: training {direction} within-class accessibility versus matched initialization; "
                        f"difference {interval(metric)}, positive in {eligible['positive_final_minus_matched_init']}/{eligible['n_defined_differences']} seeds.")
    sections.insert(sections.index("### Interpretation and limitations")+1, "\n\n".join(findings))
    return "\n\n".join(sections)


def figure(report):
    keys = ["global_trained", "global_init", "between_class_fraction", "within_trained", "within_matched_init",
            "history_shuffle", "random_input", "label_permutation"]
    labels = ["Global\ntrained", "Global\ninit", "Between\nclasses", "Within\ntrained", "Within\nmatched init",
              "History\nshuffle", "Random\ninput", "Label\npermutation"]
    colors = ["#2378a4", "#9ecae1", "#b3a2cc", "#25835c", "#a1d4b7", "#d8a24b", "#c2784a", "#bb6670"]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.7), sharey=True)
    for ax, group in zip(axes, report["aggregates"].values()):
        all_seeds, eligible = group["all_seed_descriptive"], group["within_class_eligible"]
        no_eligible = eligible["n_seeds"] == 0
        for i, key in enumerate(keys):
            source = all_seeds if i < 3 or no_eligible else eligible
            item = source["metrics"][key]
            if item["mean"] is None:
                continue
            hatch = "///" if no_eligible and i >= 3 else None
            ax.bar(i, item["mean"], color=colors[i] if not hatch else "#dddddd", edgecolor="#555555",
                   linewidth=.5, width=.73, hatch=hatch)
            if item["ci95"]:
                low, high = item["ci95"]
                ax.errorbar(i, item["mean"], yerr=[[max(0, item["mean"]-low)], [max(0, high-item["mean"])]],
                            color="#222222", capsize=2.5, linewidth=1, fmt="none")
        ax.set_xticks(range(len(keys)), labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(group["condition"]+f"\n15 seeds; conditional n={eligible['n_seeds']}; one-class={len(all_seeds['one_class_seeds'])}", fontsize=11)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set_ylim(-.12, 1.04)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if no_eligible:
            ax.text(.5, .48, "No eligible multi-class finals\nHatched bars: descriptive only", transform=ax.transAxes,
                    ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#bbbbbb"})
    axes[0].set_ylabel("Held-out R² / between-class variance fraction")
    fig.suptitle("eps02 Ladder: trained versus matched initialization inside greedy-action classes", fontsize=14, y=.98)
    fig.legend(handles=[Patch(facecolor="#dddddd", hatch="///", edgecolor="#555555",
                              label="One-class/global accessibility only — excluded from within-class claims")],
               loc="lower center", bbox_to_anchor=(.5, .065), frameon=False, fontsize=10)
    fig.text(.5, .025, "Equal-seed means; 95% CIs from 10,000 whole-seed resamples. Controls retain original targets/classes. Accessibility ≠ causal use.",
             ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.32, top=.81, wspace=.09, left=.055, right=.99)
    stem = OUTPUT.with_suffix("")
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    report = json.loads(OUTPUT.read_text())
    append_section(ROOT/"results/README.md", markdown(report))
    figure(report)
    print(f"Reported {len(report['runs'])} runs; seed means/CIs, inventory, README, PNG and vector PDF written.")


if __name__ == "__main__":
    main()
