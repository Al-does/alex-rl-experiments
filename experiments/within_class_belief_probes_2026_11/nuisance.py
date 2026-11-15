"""Verified offline entropy-0.005 persistent-nuisance conditional probes."""

import hashlib
import json
import platform
import subprocess

import jax
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.probe import fit_affine_probe, probe_predict, r2_score, global_mse_metrics
from .temp15 import ROOT, restore, histories, rows, gate_passes, TOLERANCE_REASON
from .temp15_analyze import analyze_encoder, replay, TARGETS
from .metrics import score, bootstrap

SOURCE = ROOT.parent / "belief_quotient_persistent_nuisance_jax_entropy_2026_11"
RAW = ROOT / "artifacts/nuisance_ent0p005"
OUTPUT = ROOT / "results/ladder_nuisance_ent0p005_within_class.json"
SOURCE_REF = "origin/devin/1794419502-persistent-nuisance-egreedy"


def selected_runs():
    return sorted(SOURCE.glob("variant_*_transformer_ent0p005/results/*-seed42-2000m"))


def validate_recipe(recipe):
    if (recipe["model"]["n_actions"] != 3 or len(recipe["process"]["initial"]) != 3
            or recipe["training"]["entropy_schedule"] != [[0, .005]]
            or recipe["training"]["temperature"] != 1 or recipe["training"]["epsilon"] != 0
            or recipe["seed"] != 42 or recipe["model"]["kind"] != "transformer"):
        raise ValueError("expected three-state seed-42 transformer with entropy .005, temperature 1, epsilon 0")


def archive_check(point, data):
    fit, test = data
    y = rows(test, "belief", 80000)
    w, bias = fit_affine_probe(rows(fit, "embedding", 60000), rows(fit, "belief", 60000), ridge=1e-6)
    pred = probe_predict(w, bias, rows(test, "embedding", 80000))
    actual = r2_score(pred, y)
    ratio = global_mse_metrics(pred, y)["global_mse_ratio"]
    archived = point["targets"]["belief"]
    uncertainty = bootstrap(y, pred, np.zeros(len(y), dtype=int), np.arange(len(y))//960, n=2000)
    result = {"archived": archived["r_squared"], "reproduced": actual,
              "delta": actual-archived["r_squared"],
              "archived_1_minus_r2": archived["global_mse_ratio"], "reproduced_1_minus_r2": ratio,
              "ratio_delta": ratio-archived["global_mse_ratio"], "bootstrap_se": uncertainty["r2_se"],
              "annotation_threshold": max(.01, 5*uncertainty["r2_se"]), "bootstrap": uncertainty,
              "archived_policy": point["policy"],
              "cpu_action_fractions": (np.bincount(rows(test, "action", 80000), minlength=3)/80000).tolist()}
    result["annotation"] = None if gate_passes(result) else "archived global R² not reproduced on CPU"
    return result


def analyze(runs=None, *, raw_root=RAW, validator=validate_recipe):
    if runs is None:
        runs = selected_runs()
        assert len(runs) == 3
    for run in runs:
        recipe = json.loads((run/"resolved_recipe.json").read_text())
        validator(recipe)
        output = raw_root/run.parent.parent.name/"analysis.json"
        if output.exists():
            continue
        points = json.loads((run/"condition_summary.json").read_text())["checkpoint_reports"]
        midpoint = min(points[1:-1], key=lambda p: abs(p["checkpoint_step"]-1_000_000_000))
        result = {"arm": run.parent.parent.name, "seed": 42, "stages": {}}
        init = None
        for stage, point in (("init", points[0]), ("intermediate", midpoint), ("final", points[-1])):
            step = point["checkpoint_step"]
            spec, process, params, provenance = restore(run, step, raw_root=raw_root)
            if stage == "init":
                init = params
            data = histories(run, step, spec, process, params, raw_root=raw_root)
            check = archive_check(point, data)
            classes = [rows(d, "action", n) for d, n in zip(data, (60000, 80000))]
            metrics, predictions, y = analyze_encoder(spec, params, data, [d["embedding"] for d in data], classes)
            result["stages"][stage] = {"step": step, "provenance": provenance, "archive_check": check, **metrics}
            print("CHECK", result["arm"], stage, json.dumps(check), flush=True)
            if stage == "final":
                own = np.asarray(replay(spec, params, data[1]["obs"]))
                result["final_self_replay_max_abs"] = float(np.max(np.abs(own-data[1]["embedding"])))
                assert result["final_self_replay_max_abs"] < 1e-4
                embeddings = [np.asarray(replay(spec, init, d["obs"])) for d in data]
                matched, other, _ = analyze_encoder(spec, init, data, embeddings, classes)
                result["matched_init"] = matched
                result["paired_differences"] = {}
                saved = {}
                for feature in predictions:
                    result["paired_differences"][feature] = {}
                    for target, sl in TARGETS.items():
                        p, q = predictions[feature][:, sl], other[feature][:, sl]
                        valid = np.isfinite(p).all(1) & np.isfinite(q).all(1)
                        c, episodes = classes[1][valid], (np.arange(80000)//960)[valid]
                        difference = score(y[valid, sl], p[valid], c)["r2"]-score(y[valid, sl], q[valid], c)["r2"]
                        interval = bootstrap(y[valid, sl], p[valid], c, episodes, n=10000, comparison=q[valid])
                        result["paired_differences"][feature][target] = {"difference": difference, **interval}
                    saved[feature], saved[feature+"_matched"] = predictions[feature], other[feature]
                np.savez_compressed(output.with_name("final_predictions.npz"), **saved)
                np.savez_compressed(output.with_name("matched_embeddings.npz"), fit=embeddings[0], test=embeddings[1])
            jax.clear_caches()
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print("DONE", result["arm"], flush=True)


def main():
    analyze()
    records = [json.loads((RAW/run.parent.parent.name/"analysis.json").read_text()) for run in selected_runs()]
    source_files = [*ROOT.glob("*.py"), *SOURCE.glob("*.py"),
                    *SOURCE.parent.joinpath("belief_quotient_history_jax_2026_11").glob("*.py"),
                    SOURCE.parent/"belief_quotient_persistent_nuisance_jax_2026_11/shared.py",
                    SOURCE.parent/"belief_quotient_persistent_nuisance_jax_2b_2026_11/shared.py"]
    report = {"schema_version": 1, "protocol": {
        "target": "full filtered three-state latent belief, not a marginal or product with an added nuisance factor",
        "target_dimension": 3, "delta": "b1-b2 (non-reward-state contrast)", "rho": "b3 (reward-state probability)",
        "coordinate_interpretation": "Trivial: both coordinates action-unnecessary. Reward-State: rho resolves the reward-state partition; delta is the action-irrelevant within-nonreward contrast. Full-State: both can matter for action selection; delta is not a task nuisance there.",
        "classes": "each checkpoint's own greedy action; matched init and controls retain final/source classes and targets",
        "representation": "post-final-LayerNorm embedding; 64 dimensions", "head_blind": "62-dimensional null space of two logit-difference head rows; each encoder's own head; common logit shift retained",
        "fit_rows": 60000, "test_rows": 80000, "warmup": 64, "episode_length": 1024, "ridge": 1e-6,
        "streams": "split(fold_in(PRNGKey(42),2)); greedy, study's own collector",
        "fit_episodes": 63, "test_episodes": 84, "last_scored_episode_truncated": True,
        "pooled_r2": "1 - sum(class SSE)/sum(class test-mean-centered SST); class-specific affine refits",
        "matched_init": "initialized transformer replayed on exact final-policy observation/previous-action streams, empty cache per episode, full 1024 steps including warmup; final targets/classes/scored rows retained",
        "controls": "one fixed draw per split; shuffle whole observation/action pairs except reset; random input from separate reset/nonreset empirical pair pools; fit-label permutation within source classes; original targets/classes retained",
        "control_seeds": "20261106+10*control_index+split_index; label permutation 853",
        "bootstrap": "10000 paired whole-test-episode resamples, class means recomputed, probes fixed; conditional on ONE training seed 42 and fixed fit data; 2000 draws for archive checks",
        "archive_rule": TOLERANCE_REASON, "discrepancies_are_blockers": False,
        "intermediate_selection": "saved checkpoint closest to 1B steps", "undefined": "constant/empty class R2 null; predictions excluded when class has fewer than two fit rows; counts explicit"},
        "provenance": {"source_ref": SOURCE_REF, "source_revision": subprocess.check_output(["git", "rev-parse", SOURCE_REF], text=True).strip(),
            "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "harness_commit": subprocess.check_output(["git", "-C", str(ROOT.parents[2]/"rl-harness"), "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__, "devices": str(jax.devices()),
            "source_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "import_method": "git archive of selected ent0p005 arms and small shared recipe dependencies from stacked base; no merge/rebase. Local greedy rollout equivalent at temperature 1/epsilon 0; only upstream base-collector difference is epsilon support."},
        "arms": records}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
