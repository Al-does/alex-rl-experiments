"""Refresh a saved Nonergodic Belief Explorer without refitting belief probes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from analysis.probes.controls import fit_grouped_affine, score_prediction
from analysis.simplex import (
    ExplorerScore,
    add_explorer_scores,
    write_nonergodic_belief_explorer,
)

from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.simplex_data import (
    HORIZON,
    REPO,
    ROOT,
    RUNS,
    collect_histories,
    digest,
    extract_layers,
    load_model,
)

CHECKPOINT_LABELS = {"init": "Initialization", "final": "Final"}


def archived_scores(name: str, provenance: dict) -> tuple[dict, dict, dict]:
    leaf, run_id = RUNS[name]
    layers, tasks, sources = {}, {}, {}
    for checkpoint, label in CHECKPOINT_LABELS.items():
        steps = 0 if checkpoint == "init" else int(provenance["agent_steps"])
        path = ROOT / leaf / "results" / run_id / "checkpoint_probes" / f"steps_{steps:09d}" / "probe_battery.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report["agent_steps"] != steps:
            raise ValueError("archived checkpoint step mismatch")
        sources[label] = {"path": str(path.relative_to(REPO)), "sha256": digest(path)}
        metadata = report["metadata"]
        note = (
            f"{label}: archived checkpoint evaluation, {report['n_test']:,} held-out rows; "
            f"{metadata['policy_mode']} checkpoint's own policy, seed {metadata['seed']}, "
            f"warmup {metadata['warmup_per_episode']} per episode, independent fit/test seed streams. "
            "Archived NTP predicts the pending delayed token; these histories and the original "
            "probe fitting protocol differ from the matched-history belief/log-NTP probes."
        )
        layers[label] = {}
        for site, values in report["layers"].items():
            scores = {}
            for target, title in (
                ("pending_token_distribution", "NTP"),
                ("log_pending_token_distribution", "log NTP"),
            ):
                metrics = values["probe_fits"].get(target, {})
                for metric, suffix in (("r_squared", "R²"), ("mse", "MSE")):
                    if metric in metrics:
                        scores[f"Activation → {title} {suffix} (archived)"] = ExplorerScore(metrics[metric], note)
            if scores:
                layers[label][site] = scores
        occupancy = report.get("policy", {}).get("mean_reward_state_occupancy")
        if occupancy is not None:
            tasks[f"{label} reward occupancy (archived)"] = ExplorerScore(
                occupancy,
                f"{label}: archived empirical mean reward-state occupancy on the checkpoint's own "
                f"greedy policy, {report['n_test']:,} held-out steps, seed {metadata['seed']}, "
                f"warmup {metadata['warmup_per_episode']}. No new policy evaluation.",
                "percent",
            )
    return layers, tasks, sources


def fit_missing_log_ntp(
    name: str, report: dict, source: Path, output: Path, checkpoint_names: list[str],
) -> dict:
    provenance, protocol = report["provenance"], report["protocol"]
    paths = {}
    for checkpoint, info in provenance["checkpoints"].items():
        path = REPO / info["path"]
        for filename, expected in info["files"].items():
            if digest(path / filename) != expected:
                raise ValueError(f"checkpoint hash mismatch: {name}/{checkpoint}/{filename}")
        paths[checkpoint] = path
    final = load_model(paths["final"])
    site = f"layer_{len(final.encoder.blocks)}"
    histories = {
        split: collect_histories(
            final, provenance["recipe"]["environment"],
            episodes=protocol["episodes_per_split"], seed=protocol[f"{split}_seed"],
        )
        for split in ("fit", "test")
    }
    with np.load(source / "raw" / f"{name}_heldout.npz") as saved:
        for key in ("observations", "beliefs", "tokens", "actions", "components"):
            actual = {
                "observations": histories["test"].observations, "beliefs": histories["test"].beliefs,
                "tokens": histories["test"].tokens, "actions": histories["test"].actions,
                "components": histories["test"].components,
            }[key]
            np.testing.assert_array_equal(actual, saved[key], err_msg=f"{name}: replay differs from saved {key}")
    floor = protocol["log_probability_floor"]
    targets = {
        split: np.log(np.maximum(history.pending_tokens, floor)).reshape(-1, 3)
        for split, history in histories.items()
    }
    groups = np.repeat(np.arange(protocol["episodes_per_split"]), HORIZON)
    result = {
        "site": site, "checkpoint_provenance": provenance["checkpoints"],
        "protocol": {
            "target": "natural log of pending delayed-token probabilities from the action-conditioned source filter",
            "direction": "activation -> log NTP, not log NTP -> belief",
            "log_probability_floor": floor, "sampling": protocol["sampling"],
            "fit_seed": protocol["fit_seed"], "test_seed": protocol["test_seed"],
            "analysis_seed": protocol["analysis_seed"],
            "episodes_per_split": protocol["episodes_per_split"], "positions_per_episode": HORIZON,
            "site_selection": "last residual block, before final RMSNorm; chosen before fitting, not by test scores",
            "probe": "affine; five-fold whole-episode training-only SVD-cutoff CV",
            "warmup": 0, "heldout_replay": "all saved observations, beliefs, tokens, actions and components match exactly",
            "interpretation": "linear accessibility on final-policy histories, not causal use",
        },
        "checkpoints": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoint_names:
        module = final if checkpoint == "final" else load_model(paths[checkpoint])
        features = {}
        for split, history in histories.items():
            values = extract_layers(module, history.observations, sites=(site,))
            features[split] = values[:, :, 0].reshape(len(targets[split]), -1)
        weight, bias, fit = fit_grouped_affine(
            features["fit"], targets["fit"], groups, seed=protocol["analysis_seed"],
        )
        prediction = features["test"] @ weight + bias
        path = output / f"{name}_{checkpoint}_log_ntp.npz"
        np.savez_compressed(path, prediction=prediction, target=targets["test"], weight=weight, bias=bias)
        result["checkpoints"][CHECKPOINT_LABELS[checkpoint]] = {
            "metrics": score_prediction(prediction, targets["test"]), "fit": fit,
            "raw_file": path.name, "raw_sha256": digest(path),
        }
        print(name, checkpoint, site, result["checkpoints"][CHECKPOINT_LABELS[checkpoint]]["metrics"], flush=True)
    return result


def refresh(
    source: Path, output: Path, *, supplement_path: Path | None = None,
    fit_log_ntp: bool = False,
) -> None:
    if output.exists():
        raise FileExistsError("choose a new output directory")
    if fit_log_ntp and supplement_path is None:
        raise ValueError("supply --supplement to save the new compact log-NTP results")
    data = json.loads((source / "data.js").read_text().removeprefix("window.SIMPLEX_DATA=").removesuffix(";\n"))
    reports = {item["name"]: json.loads((source / item["path"]).read_text()) for item in data["reports"]}
    supplement_exists = supplement_path is not None and supplement_path.exists()
    supplement = json.loads(supplement_path.read_text()) if supplement_exists else {
        "analysis": {
            "source_sha256": digest(Path(__file__)), "numpy": np.__version__, "torch": torch.__version__,
            "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "library_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO.parent / "rl-harness", text=True).strip(),
        },
        "runs": {},
    }
    runs = []
    for name in RUNS:
        run = next(run for run in data["runs"] if run["name"] == name.replace("_", " ").title())
        report_name = f"{name} full report"
        report = reports[report_name]
        report_path = source / next(item["path"] for item in data["reports"] if item["name"] == report_name)
        layers, tasks, sources = archived_scores(name, report["provenance"])
        if name in supplement["runs"]:
            extra = supplement["runs"][name]
            if extra["source_report_sha256"] != digest(report_path):
                raise ValueError("supplement belongs to a different source report")
        else:
            candidates = {
                site for site in [*run["sites"], *(site for values in layers.values() for site in values)]
                if site.startswith("layer_")
            }
            last_sites = sorted(candidates, key=lambda site: int(site.split("_")[1]))[-2:]
            missing = [
                checkpoint for checkpoint, label in CHECKPOINT_LABELS.items()
                if not any(
                    scores.get("Activation → log NTP R² (archived)", ExplorerScore(None, "")).value is not None
                    for site, scores in layers.get(label, {}).items() if site in last_sites
                )
            ]
            extra = {
                "source_report_sha256": digest(report_path), "archived_sources": sources,
                "bayes_maximum": "not available for alpha095; no calculation requested",
            }
            if missing and fit_log_ntp:
                extra.update(fit_missing_log_ntp(name, report, source, output / "raw", missing))
            supplement["runs"][name] = extra
        for label, values in extra.get("checkpoints", {}).items():
            scores = layers.setdefault(label, {}).setdefault(extra["site"], {})
            note = (
                f"{label}: activation → log NTP of the pending delayed token; natural logarithm, "
                f"floor {extra['protocol']['log_probability_floor']}. Matched final-policy histories, "
                f"{values['metrics']['n_evaluated']:,} held-out positions; independent complete episodes, "
                "training-only grouped SVD-cutoff CV. See Predictive score supplement for seeds and hashes."
            )
            for metric, suffix in (("r_squared", "R²"), ("mse", "MSE")):
                scores[f"Activation → log NTP {suffix}"] = ExplorerScore(values["metrics"][metric], note)
        runs.append(add_explorer_scores(run, layer_scores=layers, task_scores=tasks))
    if supplement_path is not None and not supplement_exists:
        supplement_path.parent.mkdir(parents=True, exist_ok=True)
        supplement_path.write_text(json.dumps(supplement, indent=2, allow_nan=False) + "\n")
    write_nonergodic_belief_explorer(
        output, runs, title="Nonergodic Belief Explorer · Alpha 0.95",
        description=data["description"],
        reports={**reports, "Predictive score supplement": supplement},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="existing exported viewer including raw held-out arrays")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, help="read an existing compact log-NTP report, or save a new one")
    parser.add_argument("--fit-missing-log-ntp", action="store_true", help="opt in to fitting only missing last-block log-NTP probes")
    args = parser.parse_args()
    torch.set_num_threads(2)
    refresh(args.source, args.output, supplement_path=args.supplement, fit_log_ntp=args.fit_missing_log_ntp)


if __name__ == "__main__":
    main()
