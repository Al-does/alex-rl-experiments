"""Aggregate whole-seed ladder comparisons without pooling rollout rows."""

import hashlib
import json
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .gate import ROOT
from .ladder import items, NAMES, STREAMS
from .metrics import seed_mean_intervals


def projection(result):
    return {**{f"{feature}.{target}": record["r2"]
               for feature, targets in result["metrics"].items()
               for target, record in targets.items()},
            **{f"{control}.{target}": record["r2"]
               for control, targets in result["controls"].items()
               for target, record in targets.items()}}


def aggregate(records):
    columns = sorted(records[0])
    values = np.array([[record[k] for k in columns] for record in records])
    result = seed_mean_intervals(values)
    return {"n_seeds": result["n_seeds"], "n_resamples": result["n_resamples"],
            "bootstrap_seed": result["seed"],
            "metrics": {k: {"mean": result["mean"][i], "ci95": result["ci95"][i]} for i, k in enumerate(columns)}}


def render(report):
    columns = ["full_h.belief", "head_blind.belief", "logits.belief", "full_h.delta", "full_h.rho", "history_shuffle.belief", "random_input.belief", "label_permutation.belief"]
    lines = ["## Belief-Quotient Ladder: within-greedy-action-class decoding", "",
             "Means across 15 training seeds; brackets are 95% percentile intervals from 10,000 whole-seed bootstrap draws. Full JSON includes per-seed/per-class scores, all analyzed stages, and paired trained-minus-initialized-encoder contrasts.", "",
             "| Condition | Representation/history | Full h | Head-blind | Logits | Delta | Rho | History shuffle | Random input | Label permutation |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"final": "Final trained", "init_encoder_same_rows": "Init encoder, final histories/classes", "init": "Own-policy init"}
    colors = {"final": "#2266aa", "init_encoder_same_rows": "#dd8833", "init": "#777777"}
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for index, condition in enumerate(NAMES.values()):
        data = report["aggregates"][condition]
        for group in ("final", "init_encoder_same_rows", "init"):
            metrics = data[group]["metrics"]
            text = [f"{metrics[k]['mean']:.4f} [{metrics[k]['ci95'][0]:.4f}, {metrics[k]['ci95'][1]:.4f}]" for k in columns]
            lines.append(f"| {condition} | {labels[group]} | "+" | ".join(text)+" |")
            xs = np.arange(5)+(list(labels).index(group)-1)*.23
            means = np.array([metrics[k]["mean"] for k in columns[:5]])
            ci = np.array([metrics[k]["ci95"] for k in columns[:5]])
            axes[index].errorbar(xs, means, yerr=np.stack([means-ci[:, 0], ci[:, 1]-means]), fmt="o", capsize=3, color=colors[group], label=labels[group])
        axes[index].set(title=condition, xticks=np.arange(5), xticklabels=["Full h", "Head-blind", "Logits", "Delta", "Rho"], ylabel="Mean pooled within-class R²")
        axes[index].tick_params(axis="x", rotation=25)
        axes[index].grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(ROOT / "results" / "ladder_within_class.png", dpi=180)
    plt.close(figure)
    lines += ["", "Primary paired full-h belief R² contrast (final trained minus init encoder on final histories):", "",
              "| Condition | Mean difference | 95% seed-bootstrap interval |",
              "|---|---:|---:|"]
    for condition in NAMES.values():
        contrast = report["aggregates"][condition]["paired_final_minus_init_same_rows"]["metrics"]["full_h.belief"]
        lines.append(f"| {condition} | {contrast['mean']:+.6f} | [{contrast['ci95'][0]:+.6f}, {contrast['ci95'][1]:+.6f}] |")
    lines += ["", "Primary comparison is final trained versus the seed's initialized encoder replaying exactly the final policy's histories, targets and greedy classes. Initialized head-blind/logit features use the initialized head. Own-policy initialization changes visitation and classes and is secondary. Trivial Quotient's trained policy may be constant, but initialized policies need not be; actual class counts are retained. Controls refit within the original classes against original beliefs. The empirical random control uses separate reset/non-reset pools; history shuffling preserves reset rows and complete token/previous-action pairs.", "", "These descriptive, single-campaign affine-accessibility results do not establish causal use, belief computation, or information absence. Intervals quantify between-training-seed uncertainty, not individual timestep independence."]
    path = ROOT / "results" / "README.md"
    existing = path.read_text().split("## Belief-Quotient Ladder:")[0].rstrip()
    path.write_text(existing+"\n\n"+"\n".join(lines)+"\n")


def main():
    results = []
    for item in items():
        path = item.module_path.parent / f"within_{item.stage_index}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    report = {"schema_version": 1, "protocol": {
        "fit_rows": 60000, "test_rows": 80000, "warmup": 64, "n_envs": 16, "ridge": 1e-6,
        "policy_mode": "greedy_argmax", "representation": "post_final_layer_norm", "cpu_only": True,
        "streams": STREAMS, "stream_reference": "original full-belief checkpoint probes (300/301), not later scalar campaign (510/511)",
        "classes": "unperturbed source policy greedy actions; held fixed for controls and matched initialization",
        "pooled_r2": "1-sum(class SSE)/sum(class test-mean-centred SST)",
        "head_blind": "62-dimensional null space of two logit-difference weight rows; common-shift direction retained",
        "controls": "cycle7.control_analysis shuffled_histories and empirical_random_histories; original targets and classes; fixed-ridge class refit; one draw each",
        "control_seeds": "cycle7 _CONTROL_STREAMS, child stream1 for test; label permutation seed=trainseed+811",
        "init_matched": "seed's init module replayed from reset on final-policy complete prefixes; all final fit/test rows, beliefs and classes fixed",
        "bootstrap": "15 training seeds, 10000 joint row resamples, percentile95, seed20261106; paired difference uses same seed pairing",
        "undefined": "empty/constant test classes have null R2; missing fit classes produce excluded predictions and explicit coverage counts"},
        "provenance": {"experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       "harness_commit": subprocess.check_output(["git", "-C", "/home/ubuntu/repos/rl-harness", "rev-parse", "HEAD"], text=True).strip(),
                       "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                       "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}},
        "gate": json.loads((ROOT / "results" / "ladder_reproduction_gate.json").read_text()),
        "per_seed": results, "aggregates": {}}
    for variant, name in NAMES.items():
        points = [r for r in results if r["variant"] == variant]
        stages = sorted(set(r["stage"] for r in points))
        groups = {}
        for stage in stages:
            stage_records = sorted([r for r in points if r["stage"] == stage], key=lambda r:r["seed"])
            assert [r["seed"] for r in stage_records] == list(range(42, 57)), (name, stage, len(stage_records))
            groups[stage] = aggregate([projection(r["trained_or_own_init"]) for r in stage_records])
            groups[stage]["mean_agent_steps"] = float(np.mean([r["agent_steps"] for r in stage_records]))
        final = sorted([r for r in points if r["stage"] == "final"], key=lambda r:r["seed"])
        groups["init_encoder_same_rows"] = aggregate([projection(r["init_encoder_same_rows"]) for r in final])
        contrasts = []
        for r in final:
            trained, initialized = projection(r["trained_or_own_init"]), projection(r["init_encoder_same_rows"])
            contrasts.append({k:trained[k]-initialized[k] for k in trained})
        groups["paired_final_minus_init_same_rows"] = aggregate(contrasts)
        report["aggregates"][name] = groups
    (ROOT / "results" / "ladder_within_class.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    render(report)
    print(f"Aggregated {len(results)} checkpoint reports from 45 runs; bootstrap=10000 whole-seed draws.")


if __name__ == "__main__":
    main()
