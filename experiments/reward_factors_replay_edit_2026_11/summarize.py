"""Render Markdown tables from the nine seed reports."""

from collections import Counter
import json
import statistics as stats

from .run import COHORTS, ROOT, SEEDS

SITES = ("B1", "B2", "B3", "B4", "Norm")


def fmt(vals):
    present = [x for x in vals if x is not None]
    if not present:
        return "—"
    if len(present) == 1:
        return f"{present[0]:.4f} (1 seed)"
    n = " (2 seeds)" if len(present) == 2 else ""
    return f"{stats.mean(present):.4f} ± {stats.stdev(present):.4f}{n}"


def cell(rows, get):
    return fmt([get(r) for r in rows])


def table(header, rows, trailing_blank=True):
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print("| " + " | ".join(map(str, row)) + " |")
    if trailing_blank:
        print()


def offline(joint):
    lag_cases = ("f2_0_0", "f2_1_0", "f2_8_0", "f2_32_0", "f2_128_0")
    print("## A. Factorization\n")
    rows = []
    for key in (*lag_cases, "f2_mirror", "f1_mirror"):
        group = [r["A"][key] for r in joint]
        get = lambda r: r["aligned_tv"] if "aligned_tv" in r else r["tv"]
        def at(vals, i):
            return stats.mean(r[i] for r in vals) if key.endswith("mirror") else vals[-1][i]
        rows.append([key, *(cell(group, lambda r: at(get(r), f)) for f in range(2)),
                     *(cell(group, lambda r: at(r["relative"], site)) for site in range(5)),
                     cell(group, lambda r: r["drop"])])
    overlap = [r["A"]["subspaces"] for r in joint]
    rows.append(["mirror subspace overlap", "—", "—", *(cell(overlap, lambda r: r[i]["overlap"]) for i in range(5)), "—"])
    table(("Replay", "Factor 1 TV", "Factor 2 TV", *SITES, "Mean log drop"), rows)

    print("## B. Positional replay fidelity\n")
    rows = []
    for key in ("f2_1_0", "f2_8_1", "f2_32_0", "f2_128_0", "f2_mirror"):
        for selection in ("latest", "changed", "past"):
            for factor in range(2):
                group = [r["B"][key] for r in joint]
                rows.append([key, selection, factor + 1, *(cell(group, lambda r: r[i][selection]["all"]["fidelity"][factor])
                                                           for i in range(5))])
    table(("Replay", "Positions", "Factor", *SITES), rows)

    print("## C. Factor-2 displacement prediction\n")
    rows = []
    for i, site in enumerate(SITES):
        group = [r["C"]["1"][i] for r in joint]
        models = ("False_False_none", "True_False_none", "False_True_none", "True_True_none")
        chosen = lambda r: r["models"][r["winner"]]
        def gain(r, cond):
            vals = r["models"]
            subset = [a for a in vals if a.endswith("_" + cond)]
            k = max(subset, key=lambda a: vals[a]["validation"])
            k0 = k.rsplit("_", 1)[0] + "_none"
            return vals[k]["heldout"] - vals[k0]["heldout"]
        rows.append([site, *(cell(group, lambda r: r["models"][m]["heldout"]) for m in models),
                     *(cell(group, lambda r: gain(r, c)) for c in ("other", "pc")),
                     *(cell(group, lambda r: chosen(r)[m]) for m in ("heldout", "low_drop", "high_drop", "fixed_inputs")),
                     ", ".join(f"{k} ({v})" for k, v in sorted(Counter(r["winner"] for r in group).items()))])
    table(("Site", "Prob.", "Log odds", "Prob. quadratic", "Log odds quadratic", "Gain: factor 1",
           "Gain: PCs", "Selected", "Low drop", "High drop", "Fixed latest inputs", "Selection count"), rows)

def marginal_replay(marginal):
    print("## E. Marginal Reward replay\n")
    rows = []
    for checkpoint in ("final", "initialization"):
        for key in ("f2_0_0", "f2_1_0", "f2_8_0", "f2_32_0", "f2_128_0"):
            group = [r["E"][checkpoint][key] for r in marginal]
            rows.append([checkpoint, key, cell(group, lambda r: r["tv"][-1][0]),
                         cell(group, lambda r: r["high_drop_tv"][-1][0] if r["high_drop_tv"] is not None else None),
                         *(cell(group, lambda r: r["relative"][-1][i]) for i in range(5)),
                         cell(group, lambda r: r["drop"])])
    table(("Checkpoint", "Replay", "Factor 1 TV", "High drop factor 1 TV", *SITES, "Mean log drop"), rows)


def closed_loop(group, task, heading, trailing_blank=True):
    print(heading + "\n")
    vals = [r["D"][task] for r in group]
    rows = []
    for k in ("intact", "random", "map", "mirror_input", *vals[0]["writes"]):
        g = [r[k] if k in r else r["writes"][k] for r in vals]
        def get(what, factor):
            return cell(g, lambda r: r[what][factor])
        def fidelity(f):
            return cell(g, lambda r: r.get("support", {}).get("all", {}).get("fidelity", (None, None))[f])
        row = [k, cell(g, lambda r: r["reward"]), cell(g, lambda r: r.get("gap_closed"))]
        if task == "moving":
            row += [get("components", 0), *(cell(g, lambda r: r["phase_reward"][p][1]) for p in range(2))]
        else:
            row += [get("occupancy", f) for f in range(2)]
        row += [get("change", f) for f in range(2)] + [get("exact_agree", f) for f in range(2)]
        row += [fidelity(f) for f in range(2)]
        row += [cell(g, lambda r: r.get("support", {}).get("all", {}).get("coverage"))]
        rows.append(row)
    factors = ("F1 reward", "F2 normal", "F2 reversed") if task == "moving" else ("F1 state-3", "F2 state-3")
    table(("Write", "Reward", "Gap closed", *factors, "F1 changed", "F2 changed", "F1 exact-map agree",
           "F2 exact-map agree", "F1 fidelity", "F2 fidelity", "Mirror coverage"), rows)
    archive_rows = []
    for seed, r in zip(SEEDS, vals):
        for readout in ("marginal", "joint"):
            original = r["archive"][f"discounted_{readout}_block_4"]
            rerun = r["writes"][f"{readout}_eq4_point"]["reward"]
            best = r["writes"][f"{readout}_encoder_point"]["reward"]
            reachable = r["writes"][f"{readout}_encoder_reachable"]["reward"]
            archive_rows.append([seed, readout, *(f"{x:.4f}" for x in (original["total_reward"], rerun, best, reachable))])
    table(("Seed", "Readout", "Archived Eq. (4)", "Rerun Eq. (4)", "Encoder point", "Encoder reachable"),
          archive_rows, trailing_blank)


def xor_offline(data):
    print("## XOR replay controls\n")
    rows = []
    for cohort in ("eps02", "v2"):
        for f in (0, 1):
            group = data[cohort]
            key = f"f{f + 1}_1_0"
            def r2(r, subset):
                c = r["C"][str(f)][3]
                return c["models"][c["winner"]][subset]
            rows.append([cohort, f + 1, *(cell(group, lambda r: r["A"][key]["tv"][-1][i]) for i in range(2)),
                         *(cell(group, lambda r: r["B"][key][site]["latest"]["all"]["fidelity"][f]) for site in (0, 3)),
                         *(cell(group, lambda r: r2(r, m)) for m in ("heldout", "low_drop", "high_drop"))])
    table(("Cohort", "Edited factor", "Factor 1 TV", "Factor 2 TV", "B1 latest edited-factor fidelity",
           "B4 latest edited-factor fidelity", "Block-4 R²", "Low drop R²", "High drop R²"), rows)


def main():
    data = {c: [json.loads((ROOT / "results" / f"{c}_seed{s}.json").read_text()) for s in SEEDS] for c in COHORTS}
    offline(data["eps02"])
    closed_loop(data["eps02"], "moving", "## D. Moving-reward transfer")
    marginal_replay(data["marginal"])
    xor_offline(data)
    closed_loop(data["eps02"], "xor", "## XOR: paper ε=0.02 cohort")
    closed_loop(data["v2"], "xor", "## XOR: effective v2 no-ε cohort", trailing_blank=False)


if __name__ == "__main__":
    main()
