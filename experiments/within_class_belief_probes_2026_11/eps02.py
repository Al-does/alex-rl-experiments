"""CPU-only init/final eps02 Ladder probes, controls and matched initialization."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import platform
import subprocess

import jax
import numpy as np

from harness.storage.b2 import B2StorageConfig
from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from .eps02_metrics import class_variance, summarize_seeds
from .metrics import affine, score
from .nuisance import archive_check
from .temp15 import ROOT, TOLERANCE_REASON, restore, rows
from .temp15_analyze import analyze_encoder, replay

REF = "b3db18a1cc88c0fb30c4348e3ef3b0ea6ea33ec5"
STUDIES = ["belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11",
           "belief_quotient_persistent_nuisance_eps02_jax_temp15_cont4b_2026_11"]
RAW = ROOT/"artifacts/eps02"
OUTPUT = ROOT/"results/ladder_eps02_within_class.json"


def selected_runs():
    runs = [p.parent for study in STUDIES
            for p in (ROOT.parent/study).glob("variant_*/results/*/resolved_recipe.json")]
    return sorted(runs, key=lambda p: (p.parents[2].name, p.parents[1].name,
                                      json.loads((p/"resolved_recipe.json").read_text())["seed"]))


def raw_root(run):
    return RAW/run.parents[2].name/run.name


def directory(run):
    return raw_root(run)/run.parents[1].name


def validate_recipe(recipe):
    training, model, process = recipe["training"], recipe["model"], recipe["process"]
    if (recipe["study"] not in STUDIES or recipe["environment_revision"] != "persistent_nuisance_eps02"
            or training["temperature"] != 1.5 or training["exploration_epsilon"] != 0
            or training["entropy_schedule"] != [[0, 0.]] or training["total_env_steps"] != 2_000_000_000
            or model["n_actions"] != 3 or recipe["representation"] != "post_final_layer_norm"
            or not np.allclose(process["emission"], environment.emission_matrix())
            or not np.allclose(process["transitions"], environment.action_transitions(recipe["variant"]), rtol=1e-6)
            or not np.allclose(process["initial"], [1/3]*3)
            or not np.allclose(process["reward"], [0, 0, 1])):
        raise ValueError("expected eps02 three-state Ladder, T1.5, entropy0, exploration epsilon0")


def inventory():
    client = B2StorageConfig.from_env().s3_client()

    def item(run):
        recipe = json.loads((run/"resolved_recipe.json").read_text())
        validate_recipe(recipe)
        manifest = json.loads((run/"run_manifest.json").read_text())
        summary = json.loads((run/"condition_summary.json").read_text())
        remote = manifest["remote_artifacts"]
        path = directory(run)/"durability_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            client.download_file(remote["bucket"], remote["canonical_manifest_key"], str(path))
        durable = json.loads(path.read_text())
        checkpoints = [e for e in durable["files"]
                       if e["relative_path"].startswith("checkpoints/step_") and e["relative_path"].endswith(".npz")]
        steps = sorted(int(e["relative_path"].split("step_")[1].split(".")[0]) for e in checkpoints)
        assert steps == [p["checkpoint_step"] for p in summary["checkpoint_reports"]]
        assert manifest["status"] == "completed" and remote["status"] == "completed"
        return {"study": recipe["study"], "leaf": run.parents[1].name,
                "architecture": recipe["architecture"], "variant": recipe["variant"],
                "condition": recipe["condition_name"], "seed": recipe["seed"],
                "run_id": run.name, "run_dir": str(run.relative_to(ROOT.parent)),
                "steps": steps[-1], "first_step": steps[0],
                "b2_prefix": remote["prefix"], "b2_bucket": remote["bucket"],
                "canonical_manifest_key": remote["canonical_manifest_key"],
                "durability_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "saved_checkpoints": len(steps), "checkpoint_steps": steps,
                "continued_from": recipe.get("continued_from")}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(item, selected_runs()))
    (RAW/"inventory.json").write_text(json.dumps(records, indent=2)+"\n")
    return records


def filter_histories(obs, process):
    obs = np.asarray(obs)
    initial, transitions, emission = [np.asarray(x, float) for x in
                                      (process.initial, process.transitions, process.emission)]
    if obs.ndim != 3 or obs.shape[-1] != 6 or obs.shape[1] < 1:
        raise ValueError("complete nonempty token/previous-action episode matrices required")
    tokens = obs[:, :, :3].argmax(-1)
    action = obs[:, :, 3:].argmax(-1)
    result = np.empty((*obs.shape[:2], 3))
    belief = initial[None]*emission[:, tokens[:, 0]].T
    result[:, 0] = belief/belief.sum(1, keepdims=True)
    for t in range(1, obs.shape[1]):
        belief = np.einsum("bi,bij->bj", result[:, t-1], transitions[action[:, t]])
        belief *= emission[:, tokens[:, t]].T
        result[:, t] = belief/belief.sum(1, keepdims=True)
    return result


def collect(run, step, spec, process, params):
    recipe = json.loads((run/"resolved_recipe.json").read_text())
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(recipe["seed"]), 2))
    data, validation = [], []
    for split, count, key in zip(("fit", "test"), (60000, 80000), keys):
        path = directory(run)/f"{step}_{split}.npz"
        if not path.exists():
            records = _probe_rollout(spec, process, params, key, -(-count//960), 1024)
            np.savez_compressed(path, **{k: np.asarray(records[k]) for k in
                                       ("obs", "action", "embedding", "belief", "reward")})
        with np.load(path) as saved:
            record = {k: saved[k] for k in saved.files}
        filtered = filter_histories(record["obs"], process)
        error = float(np.max(np.abs(filtered-record["belief"])))
        assert error < 3e-6, error
        assert record["belief"].shape == (*record["obs"].shape[:2], 3)
        assert rows(record, "belief", count).shape == (count, 3)
        np.testing.assert_allclose(record["belief"].sum(-1), 1, atol=3e-6)
        np.testing.assert_array_equal(record["obs"][:, 1:, 3:].argmax(-1), record["action"][:, :-1])
        validation.append({"split": split, "rows": count, "episodes": len(record["obs"]),
                           "full_history_filter_max_abs": error,
                           "history_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        data.append(record)
    return data, validation


def evaluate(spec, params, data, embeddings, classes):
    metrics, predictions, _ = analyze_encoder(spec, params, data, embeddings, classes)
    h = [rows({"h": x}, "h", n) for x, n in zip(embeddings, (60000, 80000))]
    belief = [rows(d, "belief", n) for d, n in zip(data, (60000, 80000))]
    global_prediction = affine(h[0], belief[0], h[1])
    metrics["global"] = score(belief[1], global_prediction)
    metrics["variance"] = class_variance(belief[1], classes[1], belief[0], classes[0])
    metrics["one_class"] = bool(min(np.count_nonzero(metrics["fit_counts"]),
                                    np.count_nonzero(metrics["test_counts"])) < 2)
    metrics["greedy_reward_per_step"] = float(rows(data[1], "reward", 80000).mean())
    return metrics, predictions


def analyze_run(run, *, parent=None):
    path = directory(run)/"analysis.json"
    if path.exists():
        return json.loads(path.read_text())
    recipe = json.loads((run/"resolved_recipe.json").read_text())
    validate_recipe(recipe)
    archived = json.loads((run/"condition_summary.json").read_text())["checkpoint_reports"]
    result = {"study": recipe["study"], "arm": run.parents[1].name, "run_id": run.name,
              "seed": recipe["seed"], "variant": recipe["variant"], "condition": recipe["condition_name"]}
    init_run = ROOT.parent/parent["run_dir"] if parent else run
    init_point = json.loads((init_run/"condition_summary.json").read_text())["checkpoint_reports"][0]
    init = None
    for stage, source, point in (("init", init_run, init_point), ("final", run, archived[-1])):
        step = point["checkpoint_step"]
        spec, process, params, provenance = restore(source, step, raw_root=raw_root(source))
        if stage == "init":
            init = params
        data, validation = collect(source, step, spec, process, params)
        classes = [rows(d, "action", n) for d, n in zip(data, (60000, 80000))]
        metrics, predictions = evaluate(spec, params, data, [d["embedding"] for d in data], classes)
        check = archive_check(point, data)
        result[stage] = {"step": step, "provenance": provenance, "archive_check": check,
                         "history_validation": validation, **metrics}
        print("STAGE", recipe["study"], result["arm"], recipe["seed"], stage,
              metrics["global"]["r2"], metrics["metrics"]["full_h"]["belief"]["r2"],
              "annotation", check["annotation"], flush=True)
        if stage == "final":
            own = np.asarray(replay(spec, params, data[1]["obs"]))
            result["final_self_replay_max_abs"] = float(np.max(np.abs(own-data[1]["embedding"])))
            assert result["final_self_replay_max_abs"] < 1e-4
            embeddings = [np.asarray(replay(spec, init, d["obs"])) for d in data]
            matched, other = evaluate(spec, init, data, embeddings, classes)
            result["matched_init"] = {"init_provenance": result["init"]["provenance"],
                                      "source_step": step, "source_run_id": run.name, **matched}
            assert matched["fit_counts"] == metrics["fit_counts"] and matched["test_counts"] == metrics["test_counts"]
            np.savez_compressed(directory(run)/"matched_embeddings.npz", fit=embeddings[0], test=embeddings[1])
            np.savez_compressed(directory(run)/"final_predictions.npz", **predictions,
                                **{name+"_matched": prediction for name, prediction in other.items()})
        del data, predictions
    result["within_class_eligible"] = (not result["final"]["one_class"]
        and result["final"]["metrics"]["full_h"]["belief"]["excluded_rows"] == 0)
    if parent:
        result["continued_from"] = parent
        spec, process, params, provenance = restore(run, archived[0]["checkpoint_step"], raw_root=raw_root(run))
        restored, _ = collect(run, archived[0]["checkpoint_step"], spec, process, params)
        result["continuation_join"] = {"step": archived[0]["checkpoint_step"], "provenance": provenance,
                                       "archive_check": archive_check(archived[0], restored)}
        assert provenance["checkpoint"]["sha256"] == parent["final_checkpoint_sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("DONE", result["arm"], result["seed"], result["study"], flush=True)
    return result


def assemble(items):
    records = [json.loads((directory(ROOT.parent/r["run_dir"])/"analysis.json").read_text()) for r in items]
    primary = [r for r in records if r["study"] == STUDIES[0] and r["arm"].endswith("_transformer")]
    summaries = {}
    for variant in sorted({r["variant"] for r in primary}):
        selected = [r for r in primary if r["variant"] == variant]
        summaries[str(variant)] = {"condition": selected[0]["condition"],
            "all_seed_descriptive": summarize_seeds(selected, eligible_only=False),
            "within_class_eligible": summarize_seeds(selected, eligible_only=True)}
    files = [ROOT/"eps02.py", ROOT/"eps02_metrics.py", ROOT/"eps02_report.py", ROOT/"metrics.py",
             ROOT/"temp15.py", ROOT/"temp15_analyze.py", ROOT/"nuisance.py"]
    files += [*ROOT.parent.joinpath("belief_quotient_history_jax_2026_11").glob("*.py")]
    files += [p for study in STUDIES for p in (ROOT.parent/study).glob("*.py")]
    report = {"schema_version": 1, "inventory": items, "runs": records, "aggregates": summaries,
        "protocol": {"target": "full three-state filtered belief from tokens and executed actions, recomputed with eps02 emissions; no rewards",
            "emission": environment.emission_matrix().tolist(), "delta": "b1-b2", "rho": "b3",
            "training_temperature": 1.5, "training_exploration_epsilon": 0, "training_entropy": 0,
            "checkpoints": "actual saved zero-step initialization and final; c4b uses original parent zero-step init, not the restored 2B join",
            "fit_rows": 60000, "test_rows": 80000, "episode_length": 1024, "warmup": 64, "ridge": 1e-6,
            "streams": "split(fold_in(PRNGKey(training_seed),2)); archive-compatible independent fit/test streams, reused across checkpoints within seed",
            "representation": "64D post-final-LayerNorm; head-blind null of two own-head logit-difference rows (62D); common shift retained",
            "within_class": "class-specific affine refits; 1-sum(class SSE)/sum(class test-mean-centered SST); per-class values retained",
            "between_class_fraction": "1-within-class test SST/global test SST: descriptive class-mean oracle; separate class_only_held_out fits class means on fit rows",
            "matched_init": "init encoder replayed on exact final-policy complete token/previous-action streams including warmup; final beliefs/classes/row order fixed",
            "controls": "one fixed perturbation per split, original targets/classes retained; shuffle nonreset token/action pairs within each episode; random input from empirical reset/nonreset pair pools; labels permuted within fit class; probes refit",
            "control_seeds": "20261106+10*mode_index+split_index; fit-label permutation 853",
            "bootstrap": "10000 whole-training-seed resamples, equal weight per seed; paired final-minus-matched values resampled together; no timestep pooling across seeds",
            "eligibility": "exclude finals with fewer than two observed fit/test classes or excluded full-h predictions from within-class claims; all_seed_descriptive retains them explicitly",
            "archive_rule": TOLERANCE_REASON, "archive_checks_annotation_only": True,
            "caveats": ["affine accessibility does not establish causal use or Bayesian computation",
                        "one-class within-class scores are global accessibility, not conditional evidence",
                        "CIs quantify training-seed variability conditional on fixed replay/probe streams and fit data",
                        "controls are fixed-draw descriptive distribution shifts, not significance tests",
                        "matching histories controls visitation but not causality; own-policy init is secondary",
                        "continuation restarts Adam moments; secondary comparison has one training seed"]},
        "provenance": {"source_revision": REF, "source_ref": "origin/main after git fetch origin",
            "import_method": "git archive of eps02 Ladder studies' Python sources, AGENTS and three required JSON records per run; no merge/rebase; reward_factors and unrelated main changes excluded",
            "collector_equivalence": "main differs only by epsilon-mixture training sampling/likelihood support; greedy raw-logit argmax, key splitting and belief filtering are unchanged; default probe temperature1/epsilon0 identical",
            "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "harness_commit": subprocess.check_output(["git", "-C", str(ROOT.parents[2]/"rl-harness"), "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__, "devices": str(jax.devices()),
            "source_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Wrote", OUTPUT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        parser.error("worker must be in [0, workers)")
    if any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError("CPU-only analysis")
    inventory_path = RAW/"inventory.json"
    items = (inventory() if args.inventory_only or not inventory_path.exists()
             else json.loads(inventory_path.read_text()))
    print("INVENTORY", len(items), "runs;", sum(r["saved_checkpoints"] for r in items), "durable checkpoints", flush=True)
    if args.inventory_only:
        return
    if not args.assemble_only:
        for i, item in enumerate(items):
            if i % args.workers != args.worker:
                continue
            parent = None
            if item["continued_from"]:
                parent = next(r.copy() for r in items if r["run_id"] == item["continued_from"]["run_id"])
                parent_run = ROOT.parent/parent["run_dir"]
                _, _, _, provenance = restore(parent_run, parent["steps"], raw_root=raw_root(parent_run))
                parent["final_checkpoint_sha256"] = provenance["checkpoint"]["sha256"]
            analyze_run(ROOT.parent/item["run_dir"], parent=parent)
    if args.workers == 1 or args.assemble_only:
        assemble(items)


if __name__ == "__main__":
    main()
