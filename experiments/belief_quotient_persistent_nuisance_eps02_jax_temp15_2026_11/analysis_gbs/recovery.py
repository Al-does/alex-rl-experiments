"""Recovery report from compact GBS fits, including explicitly provisional snapshots.

Example (the same command works on final published results):
  python -m experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs.recovery \
    --inputs PATH_TO_SNAPSHOT --output PATH_TO_IGNORED_REPORT/results --run-prefix gbs-v3

No fitting, remote execution, or network access occurs. Optional trusted parameter
pickles belong at <analysis-root>/artifacts/<run>/seed<seed>_hmm_k<k>.pkl, alongside
<analysis-root>/results/<run>/. Without these, recorded T/E errors and permutations
are still reported, but matrix entries are explicitly unavailable. Only the known
two-state ladder quotient and matching full-state counts are compared: no arbitrary
state-lumping search is implied. Test KL and truth never select k or model class.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from . import data
from .summarize import read_fits


CONDITIONS = {
    "variant_1": ("Trivial Quotient", 0),
    "variant_2": ("Reward-State Quotient", 1),
    "variant_3": ("Full-State Quotient", 2),
    "reward_factor_1": ("Marginal Reward", 2),
    "reward_both": ("Joint Reward", 4),
}
CLASSES = ("affine", "gru", "tanh", "hmm")
CRITERIA = {
    "elbow": "Smallest observed k with validation KL <= best validation KL + max(0.001, 0.10*abs(best validation KL)). At each k choose class by validation KL; ties use affine, gru, tanh, hmm. Apply the same rule separately within each class.",
    "adequate_fit": "Selected validation AND paired test KL <= 0.001 nats. Test is assessment only, never selection.",
    "geometry": "At the selected k, both forward and reverse MLP R2 >= 0.95 for the task target: quotient for Reward-State Quotient policy, factor_1 for Marginal Reward policy, full belief otherwise. Trivial Quotient policy at k=0 requires no belief geometry.",
    "recovered": "Complete four-class scientific k grid, adequate fit, selected k equals the stated task dimension, and geometry criterion passes.",
    "partially_recovered": "Not recovered, but adequate fit OR >=90% paired test-KL reduction relative to the validation-selected constant (k=0) predictor.",
    "not_recovered": "Neither recovery nor partial-recovery criteria established. Missing fits are pending evidence, NOT evidence of non-recoverability.",
    "condition_verdict": "Recovered only if all three policy seeds AND all three control seeds recover; partially recovered if any row recovers or partially recovers; otherwise not recovered (pending where data are absent). Any incomplete row makes the condition verdict provisional.",
    "diagnostics": "Commutation and HMM T/E agreement are descriptive, not gates: affine commutation is coordinate-dependent, and an MLP head does not ensure parameter identifiability. Quotient comparisons outside Reward-State Quotient are counterfactual.",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def winner(rows):
    candidates = [row for row in rows if finite(row.get("val_kl"))]
    return min(candidates, key=lambda row: (row["val_kl"], CLASSES.index(row["class"]))) if candidates else None


def elbow(curve):
    available = [row for row in curve if finite(row.get("val_kl"))]
    if not available:
        return None
    best = min(row["val_kl"] for row in available)
    threshold = best+max(.001, .10*abs(best))
    return min((row for row in available if row["val_kl"] <= threshold), key=lambda row: row["k"])


def selected_metrics(row):
    if row is None:
        return None
    fit = row["fit"]
    return {key: row[key] for key in ("class", "k", "val_kl", "test_kl")} | {
        "width": fit["selected"].get("width"), "selected_update": fit["selected"].get("selected_update"),
        "fit_status": fit.get("status"), "source_file": str(row["path"]),
        "geometry": fit.get("geometry"), "commutation": fit.get("commutation"),
        "probe_errors": {key: value for key, value in fit.items() if key.endswith("_error")},
    }


def matrix_detail(row, trust_parameters):
    fit, path = row["fit"], row["path"]
    result = {"k": row["k"], **(fit.get("hmm_matrices") or {"applicable": False, "reason": "comparison missing or failed"})}
    if not result.get("applicable"):
        return result
    result["counterfactual_reference"] = result.get("reference") == "exact_reward_state_quotient" and fit["condition"] != "variant_2"
    parameter = path.parents[2]/"artifacts"/path.parent.name/f"seed{fit['seed']}_hmm_k{fit['k']}.pkl"
    if not trust_parameters or not parameter.exists():
        result.update(matrices_available=False, matrices_reason="Supply matching trusted parameter artifacts and --trusted-parameters for matrix entries; recorded permutation/MSE remain available.")
        return result
    try:
        with parameter.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["spec"]["k"] != fit["k"] or saved["spec"]["kind"] != "hmm":
            raise ValueError("parameter specification differs from fit")
        def softmax(value):
            value = np.asarray(value, dtype=np.float64)
            value = np.exp(value-value.max(-1, keepdims=True))
            return value/value.sum(-1, keepdims=True)
        T, E = softmax(saved["params"]["T"]), softmax(saved["params"]["E"])
        source = (data.ROOT.parent/fit["checkpoint"]["results_path"]/"resolved_recipe.json").resolve()
        if not source.is_relative_to(data.ROOT.parent):
            raise ValueError("source recipe lies outside repository")
        process = json.loads(source.read_text())["process"]
        true_T, true_E = np.asarray(process["transitions"]), np.asarray(process["emission"])
        if result["reference"] == "exact_reward_state_quotient":
            from ..environment import action_transitions, emission_matrix
            fine_T, fine_E = action_transitions(2), emission_matrix()
            true_T = np.stack((fine_T[:, [0, 2], :2].sum(-1), fine_T[:, [0, 2], 2]), -1)
            true_E = np.stack((fine_E[[0, 2], :2].sum(-1), fine_E[[0, 2], 2]), -1)
            E = np.stack((E[:, :2].sum(-1), E[:, 2]), -1)
            result["reference_state_indices"] = [[0, 1], [2]]
            result["merged_token_indices"] = [[0, 1], [2]]
        order = result["permutation"]
        if sorted(order) != list(range(len(E))):
            raise ValueError("invalid recorded permutation")
        T, E = T[:, order][:, :, order], E[order]
        if T.shape != true_T.shape or E.shape != true_E.shape or not all(np.isfinite(a).all() for a in (T, E, true_T, true_E)):
            raise ValueError("nonfinite or mismatched matrices")
        errors = (float(np.square(T-true_T).mean()), float(np.square(E-true_E).mean()))
        result.update(matrices_available=True, parameter_sha256=hashlib.sha256(parameter.read_bytes()).hexdigest(),
                      learned_T_aligned=T.tolist(), learned_E_aligned=E.tolist(), true_T=true_T.tolist(), true_E=true_E.tolist(),
                      recomputed_transition_mse=errors[0], recomputed_emission_mse=errors[1],
                      recorded_errors_agree=bool(np.allclose(errors, [result["transition_mse"], result["emission_mse"]], atol=1e-7, rtol=1e-4)))
    except Exception as error:
        result.update(matrices_available=False, matrices_reason=f"{type(error).__name__}: {error}")
    return result


def assess(group, selected, constant):
    if selected is None:
        return {"verdict": "not recovered", "assessment": "pending: no fit evidence", "adequate_fit": False,
                "dimension_matches": None, "geometry_passes": None, "test_kl_reduction": None}
    adequate = all(finite(selected[key]) and selected[key] <= .001 for key in ("val_kl", "test_kl"))
    dimension = selected["k"] == group["true_dimension"]
    target = "belief" if group["control"] else {"variant_2": "quotient", "reward_factor_1": "factor_1"}.get(group["condition"], "belief")
    metrics = (selected.get("geometry") or {}).get(target) or {}
    geometry = all(finite(metrics.get(key)) and metrics[key] >= .95 for key in ("mlp_r2", "reverse_mlp_r2"))
    if group["condition"] == "variant_1" and not group["control"] and selected["k"] == 0:
        geometry = True
    reduction = None
    if constant and finite(constant.get("test_kl")) and constant["test_kl"] > 1e-12 and finite(selected["test_kl"]):
        reduction = 1-selected["test_kl"]/constant["test_kl"]
    recovered = group["grid_complete"] and adequate and dimension and geometry
    verdict = "recovered" if recovered else "partially recovered" if adequate or (reduction is not None and reduction >= .9) else "not recovered"
    return {"verdict": verdict, "assessment": "complete-grid assessment" if group["grid_complete"] else "provisional: incomplete sweep",
            "adequate_fit": adequate, "dimension_matches": dimension, "geometry_target": target,
            "geometry_passes": geometry, "test_kl_reduction": reduction}


def build_report(inputs, *, run_prefix="gbs-v3", trust_parameters=False):
    records, runs = {}, {}
    for manifest, meta, path, fit in read_fits(inputs, include_partial=True, run_prefix=run_prefix):
        if fit["condition"] not in CONDITIONS or fit["class"] not in CLASSES or fit["seed"] not in (42, 43, 44):
            raise ValueError(f"unexpected scientific grid at {path}")
        key = (fit["condition"], fit["seed"], fit["control"])
        records.setdefault(key, []).append({"k": fit["k"], "class": fit["class"], "val_kl": fit["selected"].get("val_kl"),
                                            "test_kl": fit.get("test_kl"), "fit": fit, "path": path})
        runs[str(manifest)] = {"run_id": manifest.parent.name, "status": meta["status"],
                               "commit": meta.get("git", {}).get("commit"), "config": meta.get("config"),
                               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    groups = []
    for condition, (label, dimension) in CONDITIONS.items():
        ks = list(range(7 if condition.startswith("variant_") else 11))
        for control in (False, True):
            for seed in (42, 43, 44):
                rows = records.get((condition, seed, control), [])
                if any(row["k"] not in ks for row in rows):
                    raise ValueError("dimension outside scientific grid")
                group = {"condition": condition, "label": label, "seed": seed, "control": control,
                         "true_dimension": (2 if condition.startswith("variant_") else 4) if control else dimension,
                         "fits_available": len(rows), "fits_expected": len(CLASSES)*len(ks),
                         "grid_complete": len(rows) == len(CLASSES)*len(ks) and all(row["fit"].get("status") == "completed" and finite(row["val_kl"]) for row in rows)}
                classes = {}
                for kind in CLASSES:
                    curve = [next((row for row in rows if row["class"] == kind and row["k"] == k),
                                  {"k": k, "class": kind, "val_kl": None, "test_kl": None}) for k in ks]
                    classes[kind] = {"curve": [{key: row[key] for key in ("k", "val_kl", "test_kl")} for row in curve],
                                     "selected": selected_metrics(elbow(curve))}
                envelope, selection_rows = [], []
                for k in ks:
                    at_k = [row for row in rows if row["k"] == k]
                    best = winner(at_k)
                    if best:
                        selection_rows.append(best)
                    tests = [row["test_kl"] for row in at_k if finite(row["test_kl"])]
                    envelope.append({"k": k, "class": best["class"] if best else None,
                                     "val_kl": best["val_kl"] if best else None, "test_kl": best["test_kl"] if best else None,
                                     "descriptive_min_test_kl": min(tests) if tests else None,
                                     "classes_available": len(at_k)})
                selected = selected_metrics(elbow(selection_rows))
                group.update(classes=classes, envelope=envelope, selected=selected,
                             best_val_kl=min((row["val_kl"] for row in selection_rows), default=None),
                             hmm_comparisons=[matrix_detail(row, trust_parameters) for row in sorted(rows, key=lambda row: row["k"]) if row["class"] == "hmm"])
                constant = next((row for row in envelope if row["k"] == 0 and row["class"] is not None), None)
                group.update(assess(group, selected, constant))
                groups.append(group)
    verdicts = []
    for condition, (label, _) in CONDITIONS.items():
        subset = [group for group in groups if group["condition"] == condition]
        values = [group["verdict"] for group in subset]
        verdict = "recovered" if all(v == "recovered" for v in values) else "partially recovered" if any(v != "not recovered" for v in values) else "not recovered"
        verdicts.append({"condition": condition, "label": label, "verdict": verdict,
                         "provisional": not all(group["grid_complete"] for group in subset),
                         "recovered_rows": values.count("recovered"), "assessed_rows": sum(group["selected"] is not None for group in subset), "expected_rows": 6})
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "run_prefix": run_prefix,
            "criteria": CRITERIA, "source_runs": runs, "groups": groups, "condition_verdicts": verdicts,
            "caveats": ["Missing fits are not imputed. All incomplete-sweep selections and verdicts are provisional.",
                        "Policy imitation dimension need not equal environmental task-sufficient dimension: the trained policy may retain history/nuisance dependence.",
                        "KL is nats per position (controls average equally over current actions); every fit uses complete held-out episodes.",
                        "R2 and commutation are descriptive held-out probes, not evidence of causal use. No uncertainty is invented from aggregate metrics.",
                        "The descriptive minimum test KL across classes is not used in any selection."]}


def number(value):
    return f"{value:.4g}" if finite(value) else "—"


def markdown(report):
    lines = ["# GBS recovery report", "", f"Generated: {report['generated_at']}", "",
             "## Criteria and limitations", ""]
    lines += [f"- **{name.replace('_', ' ')}:** {text}" for name, text in report["criteria"].items()]
    lines += [f"- {text}" for text in report["caveats"]]
    lines += ["", "## Condition verdicts", ""]
    for row in report["condition_verdicts"]:
        lines.append(f"- **{row['label']}: {row['verdict']}** ({'provisional; ' if row['provisional'] else ''}{row['recovered_rows']}/6 rows recovered; {row['assessed_rows']}/6 have fit evidence).")
    lines += ["", "## Seed-level recovery", "", "Provisional/not-recovered pending rows are not negative scientific conclusions.", "",
              "| Condition | Mode | Seed | Fits | k / true | Class | Val KL | Test KL | Verdict / assessment |",
              "|---|---|---:|---:|---:|---|---:|---:|---|"]
    for group in report["groups"]:
        selected = group["selected"] or {}
        lines.append(f"| {group['label']} | {'control' if group['control'] else 'policy'} | {group['seed']} | {group['fits_available']}/{group['fits_expected']} | {selected.get('k', '—')} / {group['true_dimension']} | {selected.get('class', '—')} | {number(selected.get('val_kl'))} | {number(selected.get('test_kl'))} | {group['verdict']}; {group['assessment']} |")
    lines += ["", "## Validation envelopes", "", "![KL versus k](kl_vs_k.pdf)", "",
              "One curve per seed and mode; at each k the class is selected by validation KL. Negative roundoff KL is clipped to zero for plotting only. Missing k are gaps. Vertical lines mark policy/control task dimensions.", "",
              "## Curves and selected-state diagnostics", "", "Each curve cell is validation / paired test KL. Best is the validation-selected class, not a test-selected model. Full per-class selected geometry and aligned matrix entries are in recovery.json."]
    for group in report["groups"]:
        if not group["fits_available"]:
            continue
        lines += ["", f"### {group['label']} — {'control' if group['control'] else 'policy'}, seed {group['seed']}", "",
                  "| k | Affine | GRU | Tanh | HMM | Best class: val / test |", "|---:|---|---|---|---|---|"]
        for i, best in enumerate(group["envelope"]):
            cells = [f"{number(group['classes'][kind]['curve'][i]['val_kl'])} / {number(group['classes'][kind]['curve'][i]['test_kl'])}" for kind in CLASSES]
            lines.append(f"| {best['k']} | "+" | ".join(cells)+f" | {best['class'] or '—'}: {number(best['val_kl'])} / {number(best['test_kl'])} |")
        lines += ["", "Class-specific elbows: "+", ".join(f"{kind} k={value['selected']['k'] if value['selected'] else '—'}" for kind, value in group["classes"].items())+"."]
        selected = group["selected"] or {}
        lines += ["", "Selected commutation: `"+json.dumps(selected.get("commutation"), allow_nan=False)+"`.",
                  "", "| Target | g→truth affine R² | g→truth MLP R² | truth→g affine R² | truth→g MLP R² |", "|---|---:|---:|---:|---:|"]
        for target, metrics in (selected.get("geometry") or {}).items():
            lines.append(f"| {target} | "+" | ".join(number(metrics.get(key)) for key in ("affine_r2", "mlp_r2", "reverse_affine_r2", "reverse_mlp_r2"))+" |")
        if selected.get("probe_errors"):
            lines.append("Probe errors: `"+json.dumps(selected["probe_errors"])+"`.")
        lines += ["", "HMM T/E comparisons (recorded best shared state permutation; only known exact quotient lumping):", "",
                  "| k | Reference | Transition MSE | Emission MSE | Matrix entries |", "|---:|---|---:|---:|---|"]
        for matrix in group["hmm_comparisons"]:
            reference = matrix.get("reference", matrix.get("reason", "unavailable"))
            if matrix.get("counterfactual_reference"):
                reference += " (counterfactual)"
            lines.append(f"| {matrix['k']} | {reference} | {number(matrix.get('transition_mse'))} | {number(matrix.get('emission_mse'))} | {'in JSON' if matrix.get('matrices_available') else matrix.get('matrices_reason', 'not applicable')} |")
    return "\n".join(lines)+"\n"


def make_figure(report):
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.2))
        for ax, (condition, (label, dimension)) in zip(axes.flat, CONDITIONS.items()):
            subset = [group for group in report["groups"] if group["condition"] == condition]
            for group in subset:
                color = "#2463a5" if not group["control"] else "#cf6c19"
                curve = group["envelope"]
                ax.plot([row["k"] for row in curve], [max(0., row["val_kl"]) if finite(row["val_kl"]) else np.nan for row in curve],
                        color=color, linestyle={42: "-", 43: "--", 44: ":"}[group["seed"]], linewidth=1, marker=".", markersize=3)
            control_dim = 2 if condition.startswith("variant_") else 4
            ax.axvline(dimension, color="#2463a5", linestyle="--", linewidth=.8, alpha=.7)
            ax.axvline(control_dim, color="#cf6c19", linestyle=":", linewidth=1, alpha=.8)
            ax.set_title(label+f"\ntrue d: policy {dimension}, control {control_dim}")
            ax.set(xlabel="k", ylabel="Validation KL (nats)", xlim=(-.25, 6.25 if condition.startswith("variant_") else 10.25))
            ax.set_xticks(range(0, 7 if condition.startswith("variant_") else 11, 2))
            ax.set_yscale("symlog", linthresh=1e-4)
            values = [row["val_kl"] for group in subset for row in group["envelope"] if finite(row["val_kl"])]
            ax.set_ylim(0, max(.001, 1.1*max(values, default=0)))
            ax.grid(alpha=.2)
        axes.flat[-1].axis("off")
        handles = [Line2D([], [], color=color, label=label) for color, label in (("#2463a5", "Policy"), ("#cf6c19", "Control"))]
        handles += [Line2D([], [], color=".3", linestyle=style, label=f"Seed {seed}") for seed, style in ((42, "-"), (43, "--"), (44, ":"))]
        axes.flat[-1].legend(handles=handles, loc="center", frameon=False)
        partial = any(not group["grid_complete"] for group in report["groups"])
        fig.suptitle("GBS validation envelopes"+(" — partial data" if partial else ""), fontsize=9)
        fig.subplots_adjust(left=.09, right=.98, bottom=.12, top=.82, hspace=.7, wspace=.55)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", type=Path, default=[data.ROOT/study/"analysis_gbs"/"results" for study in (data.LADDER, data.RR)])
    parser.add_argument("--output", type=Path, default=data.ROOT/data.LADDER/"analysis_gbs"/"results"/"gbs-v3-recovery")
    parser.add_argument("--run-prefix", default="gbs-v3")
    parser.add_argument("--trusted-parameters", action="store_true", help="load matching trusted local parameter pickles (pickle can execute code)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(args.inputs, run_prefix=args.run_prefix, trust_parameters=args.trusted_parameters)
    args.output.mkdir(parents=True)
    (args.output/"recovery.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.output/"recovery.md").write_text(markdown(report))
    fig = make_figure(report)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(args.output/"kl_vs_k.pdf", bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    print(f"wrote {len(report['groups'])} seed/mode rows, {sum(g['fits_available'] for g in report['groups'])} fits to {args.output}")


if __name__ == "__main__":
    main()
