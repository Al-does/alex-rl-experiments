"""Print the compact tables from the six seed reports."""

import itertools
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
CONDITIONS = {2: "Reward-State", 3: "Full-State"}


def stat(values, digits=3):
    xs = [x for x in values if x is not None]
    if not xs:
        return "—"
    mean = f"{statistics.mean(xs):.{digits}f}"
    if len(xs) > 1:
        mean += f" ± {statistics.stdev(xs):.{digits}f}"
    return mean if len(xs) == 3 else f"{mean} (n={len(xs)})"


def row(*cells):
    print("| " + " | ".join(map(str, cells)) + " |")


def average(xs):
    return statistics.mean(xs[8:])


def part_a(by):
    print("### A. Invariance\n")
    row("Condition", "Checkpoint", "Replay", "Policy TV", "Greedy changed", "Block 4 rel. disp.", "Norm rel. disp.",
        "Full ρ MAE", "Merged C posterior MAE", "Drop (nat/step)")
    row(*(["---"] * 10))
    for variant, checkpoint, replay in itertools.product(by, ("init", "final"), ("swap", "relabel", "flip")):
        a = [r["A"][checkpoint][replay] for r in by[variant]]
        values = [stat([average(r[k]) for r in a], 4) for k in ("tv", "greedy")]
        values += [stat([average([p[site] for p in r["relative"]]) for r in a], 4) for site in (3, 4)]
        values += [stat([r["belief_mae"][index] for r in a], 4) for index in (2, 3)]
        row(CONDITIONS[variant], checkpoint, replay, *values, stat([average(r["drop"]) for r in a], 4))
    print()
    for variant, reports in by.items():
        a = [r["A"]["final"] for r in reports]
        print(f"- {CONDITIONS[variant]} relabel Δρ→Δlogit held-out R²: "
              f'{stat([r["relabel"]["rho_logit_r2"] for r in a])}.')
        print(f"- {CONDITIONS[variant]} block-4 probes, R² (ρ, signed δ, |δ|, merged C posterior): "
              + "; ".join(stat([r["probe"][3][i] for r in a]) for i in range(4)) + ".")
    for checkpoint in ("init", "final"):
        print(f"- Full-State {checkpoint} swap TV after aligning Tilt X and Tilt Y: "
              f'{stat([average(r["A"][checkpoint]["swap"]["symmetry_tv"]) for r in by[3]], 4)}.')
    print()


def part_b(by):
    print("### B. Positional patching\n")
    cols = ("latest", "changed", "past")
    row("Condition", "Lag(s)", "Replay TV", *(f"Block 1 {col}" for col in cols), *(f"Block 4 {col}" for col in cols))
    row(*(["---"] * 9))
    for variant, reports in by.items():
        for case in ("0_0", "1_0", "2_0", "4_0", "8_0", "16_0", "32_0", "64_0", "128_0", "8_1", "32_1"):
            lag, joint = case.split("_")
            label = f"{lag}, 0" if joint == "1" else lag
            result = [r["B"][case] for r in reports]
            row(CONDITIONS[variant], label, stat([r[0]["latest"]["all"]["replay_tv"] for r in result], 6),
                *(stat([r[i][col]["all"]["fidelity"] if r[i][col]["all"]["replay_tv"] >= 1e-6 else None
                        for r in result]) for i in (0, 3) for col in cols))
    print()


def part_c(by):
    print("### C. Encoding displacement\n")
    row("Condition", "Site", "Selection winners (seeds 42, 43, 44)", "Held-out R²", "Held-out R², fixed token",
        "Best c family", "Best ρ family", "Best ρ+δ family", "Best ρ+|δ| family", "PC ΔR²", "High-drop pairs")
    row(*(["---"] * 11))
    for variant, reports in by.items():
        for site in range(5):
            models = [r["C"][site] for r in reports]
            winners = [m["winner"] for m in models]
            scores = [m["models"][w] for m, w in zip(models, winners)]
            def best(key):
                families = [{name: fit for name, fit in m["models"].items() if name.rsplit("_", 3)[0] == key} for m in models]
                return stat([max(group.values(), key=lambda x: x["selection_r2"])["heldout_r2"] for group in families])
            row(CONDITIONS[variant], site + 1 if site < 4 else "Norm",
                ", ".join(w.replace("_False", "−").replace("_True", "+") for w in winners),
                stat([x["heldout_r2"] for x in scores]), stat([x["heldout_fixed_token_r2"] for x in scores]),
                *(best(k) for k in ("c", "rho", "rho_delta", "rho_abs_delta")),
                stat([m["models"][w.rsplit("_", 1)[0] + "_True"]["heldout_r2"] -
                      m["models"][w.rsplit("_", 1)[0] + "_False"]["heldout_r2"] for m, w in zip(models, winners)]),
                stat([m["high_drop_rows"] / m["total_rows"] for m in models]))
    print()


def part_d(by):
    print("### D. Closed-loop Reward-State writes\n")
    row("Site", "Edit", "Write", "λ", "Occupancy", "Actions changed", "Fidelity", "Fidelity (high drop)", "Replay ρ MAE")
    row(*(["---"] * 9))
    reports = [r["D"] for r in by[2]]
    for site_label in ("Block 4", "B-selected"):
        ids = [3 if site_label == "Block 4" else r["supported_site"] for r in reports]
        if site_label == "B-selected" and ids == [3, 3, 3]:
            continue
        for goal, method in itertools.product(("rho_flip", "rho_point", "mirror"),
                                               ("eq4", "encoder_exact", "encoder_decoded", "reference")):
            runs = [r["sites"][str(i)][goal][method] for r, i in zip(reports, ids)]
            vals = [r["confirmation"][str(r["strength"])] for r in runs]
            row(site_label, goal, method, stat([r["strength"] for r in runs]),
                *(stat([v[k] for v in vals]) for k in ("occupancy", "action_change")),
                *(stat([v["fidelity"][k]["fidelity"] for v in vals]) for k in ("all", "high_drop")),
                stat([v["replay_rho_mae"] for v in vals]))
    for goal in ("intact", "random", "rho_flip", "rho_point", "mirror"):
        ref = reports[0]["exact_filter_reference"][goal]
        row("—", goal, "Exact-filter grid policy", "—", f'{ref["occupancy"]:.3f}', f'{ref["action_change"]:.3f}',
            "—", "—", "—")
    print()
    for control in ("intact", "random"):
        print(f'- Network {control} occupancy: {stat([r["sites"]["3"][control]["occupancy"] for r in reports])}.')
    sites = ["Final norm" if (i := r["supported_site"]) == 4 else f"Block {i + 1}" for r in reports]
    print(f"- B-selected sites (42–44): {', '.join(sites)}.")


def main():
    by = {variant: [json.loads((ROOT / "results" / f"v{variant}_seed{seed}.json").read_text())
                    for seed in (42, 43, 44)] for variant in (2, 3)}
    for part in (part_a, part_b, part_c, part_d):
        part(by)


if __name__ == "__main__":
    main()
