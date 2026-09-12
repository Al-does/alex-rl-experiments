"""Matched-history, action-conditioned simplex probes for the PR 122 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from analysis.belief_geometry import evaluate_belief_geometry
from analysis.checkpoints import load_module_only
from analysis.simplex import build_simplex_run, write_simplex_viewer
from analysis.simplex import geometry_metrics as score_geometry
from envs.hmm import HMMEnv
from harness.storage.b2 import B2StorageConfig

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.analysis import (
    ActionConditionedTransducerTarget,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RUNS = {
    "variant_2": ("alpha095/variant_2", "nem3-rsa-c5-alpha095-variant_2-s42"),
    "variant_3": ("alpha095/variant_3", "nem3-rsa-c5-alpha095-variant_3-s42"),
}
HORIZON = 128
SITES = ("layer_1", "layer_2", "layer_3", "layer_4", "post_final_norm")
COMPONENT_INDICES = {"Component A": [0, 1, 2], "Component B": [3, 4, 5]}
STUDY_DESCRIPTION = (
    "Alpha095 variants 2 and 3: actual initialization and final checkpoints from PR 122. "
    "Both checkpoints see identical final-policy greedy histories; initialization is an "
    "off-policy representation control. Independent complete episodes form the fit and test sets. "
    "Post-final RMSNorm is primary; all four pre-normalization residual streams are robustness comparisons.\n\n"
    "The target is the decision-time arrival belief, conditioned on visible delayed tokens and "
    "executed actions, never reward or latent state. Position 0 is BOS; positions 1–127 reveal "
    "delayed tokens. Position 127 is terminal, with no next action. The independent filter is "
    "checked against public diagnostics at every row.\n\n"
    "Affine probes use training-only whole-episode cross-validation. Reports include predictive "
    "and current-observation controls, initialization, Gaussian and shuffled-label nulls, "
    "target variance and episode-bootstrap comparisons. One training seed per run limits "
    "generalization across trained models. Different variants have different visitation."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_run(name: str) -> tuple[dict, dict[str, Path]]:
    leaf, run_id = RUNS[name]
    source = ROOT / leaf / "results" / run_id
    manifest = json.loads((source / "run_manifest.json").read_text())
    recipe = json.loads((source / "resolved_recipe.json").read_text())
    trial = json.loads((source / "tune_summary.json").read_text())["trials"][0]
    remote = manifest["remote_artifacts"]
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 credentials are required to restore the saved checkpoints")
    client = config.s3_client()
    raw = client.get_object(
        Bucket=remote["bucket"], Key=remote["canonical_manifest_key"],
    )["Body"].read()
    durability = json.loads(raw)
    final_parts = Path(trial["checkpoint"]).parts
    final_relative = Path(*final_parts[final_parts.index(run_id) + 1:])
    suffix = Path("learner_group/learner/rl_module/default_policy")
    roots = {"init": Path("initial_checkpoint") / suffix, "final": final_relative / suffix}
    local = ROOT / leaf / "artifacts" / run_id
    provenance = {
        "run_id": run_id,
        "training_git": manifest["git"],
        "recipe": recipe,
        "agent_steps": trial["metrics"]["env_runners/num_env_steps_sampled_lifetime"],
        "durability_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoints": {},
    }
    paths = {}
    for checkpoint, prefix in roots.items():
        files = [
            entry for entry in durability["files"]
            if Path(entry["relative_path"]).parent == prefix
        ]
        if {Path(entry["relative_path"]).name for entry in files} != {
            "class_and_ctor_args.pkl", "metadata.json", "module_state.pkl",
        }:
            raise ValueError(f"unexpected module subtree for {name}/{checkpoint}")
        for entry in files:
            destination = local / entry["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or digest(destination) != entry["sha256"]:
                client.download_file(remote["bucket"], entry["key"], str(destination))
            if destination.stat().st_size != entry["size_bytes"] or digest(destination) != entry["sha256"]:
                raise ValueError(f"checkpoint checksum mismatch: {destination}")
        paths[checkpoint] = local / prefix
        provenance["checkpoints"][checkpoint] = {
            "path": str(paths[checkpoint].relative_to(REPO)),
            "files": {Path(entry["relative_path"]).name: entry["sha256"] for entry in files},
        }
    return provenance, paths


def load_model(path: Path) -> FactoredReproductionActorCritic:
    model = load_module_only(path)
    if not isinstance(model, FactoredReproductionActorCritic):
        raise TypeError("expected the saved FactoredReproductionActorCritic")
    return model.cpu().eval()


@dataclass
class Histories:
    observations: np.ndarray
    beliefs: np.ndarray
    pending_tokens: np.ndarray
    tokens: np.ndarray
    actions: np.ndarray
    components: np.ndarray
    diagnostic_error: float


@torch.inference_mode()
def collect_histories(
    model: FactoredReproductionActorCritic,
    environment: dict,
    *,
    episodes: int,
    seed: int,
    batch_size: int = 16,
) -> Histories:
    if episodes < 2 or batch_size < 1:
        raise ValueError("need at least two episodes and a positive batch size")
    config = {
        **environment,
        "diagnostics": {"belief": True, "state": True, "tokens": True, "transitions": True},
    }
    if config["episode_length"] != HORIZON - 1:
        raise ValueError("expected 127 actions plus the reset observation")
    seeds = np.random.SeedSequence(seed).spawn(episodes)
    output = {key: [] for key in ("observations", "beliefs", "pending_tokens", "tokens", "actions", "components")}
    diagnostic_error = 0.0
    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        envs = [HMMEnv(config) for _ in range(count)]
        try:
            reset = [env.reset(seed=int(seeds[start + i].generate_state(1)[0])) for i, env in enumerate(envs)]
            observations, infos = zip(*reset)
            target = ActionConditionedTransducerTarget(envs[0].model, count)
            obs = np.zeros((count, HORIZON, envs[0].observation_space.shape[0]), dtype=np.float32)
            beliefs = np.zeros((count, HORIZON, 6), dtype=np.float64)
            pending = np.zeros((count, HORIZON, 3), dtype=np.float64)
            tokens = np.full((count, HORIZON), -1, dtype=np.int64)
            actions = np.full((count, HORIZON), -1, dtype=np.int64)
            components = np.array([info["state_current"] // 3 for info in infos])
            for step in range(HORIZON):
                obs[:, step] = np.stack(observations)
                targets = target(obs[:, step], infos, np.full(count, step))
                beliefs[:, step] = targets["weighted_belief"]
                pending[:, step] = targets["pending_token_distribution"]
                tokens[:, step] = targets["token"]
                diagnostic_error = max(
                    diagnostic_error,
                    float(np.abs(beliefs[:, step] - targets["diagnostic_belief"]).max()),
                )
                if not np.array_equal(targets["state"] // 3, components):
                    raise AssertionError("a trajectory crossed nonergodic components")
                if step == HORIZON - 1:
                    break
                residual = model.encoder.forward_complete_episode(torch.from_numpy(obs[:, :step + 1]))
                logits = model.action_distribution_inputs(residual[:, -1])
                actions[:, step] = logits.argmax(dim=-1).numpy()
                results = [env.step(int(action)) for env, action in zip(envs, actions[:, step])]
                observations = [row[0] for row in results]
                infos = [row[4] for row in results]
                if any(bool(row[2] or row[3]) != (step == HORIZON - 2) for row in results):
                    raise AssertionError("unexpected episode boundary")
            for key, value in (
                ("observations", obs), ("beliefs", beliefs), ("pending_tokens", pending),
                ("tokens", tokens), ("actions", actions), ("components", components),
            ):
                output[key].append(value)
        finally:
            for env in envs:
                env.close()
        print(f"histories seed={seed}: {start + count}/{episodes}", flush=True)
    if diagnostic_error > 1e-10:
        raise AssertionError(f"independent filter disagrees with public diagnostics: {diagnostic_error}")
    return Histories(**{key: np.concatenate(value) for key, value in output.items()}, diagnostic_error=diagnostic_error)


@torch.inference_mode()
def extract_layers(model: FactoredReproductionActorCritic, observations: np.ndarray) -> np.ndarray:
    captured = {}

    def capture(layer: int):
        def hook(_module, _inputs, output):
            captured[layer] = output.detach().numpy().copy()
        return hook

    handles = [block.register_forward_hook(capture(i)) for i, block in enumerate(model.encoder.blocks)]
    batches = []
    try:
        for start in range(0, len(observations), 32):
            captured.clear()
            normalized = model.encoder.forward_complete_episode(torch.from_numpy(observations[start:start + 32]))
            batches.append(np.stack(
                [captured[i] for i in range(len(handles))] + [normalized.numpy().copy()],
                axis=2,
            ))
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(batches)


def geometry_metrics(prediction: np.ndarray, beliefs: np.ndarray) -> dict:
    return score_geometry(prediction, beliefs, components=COMPONENT_INDICES)


def analyze_run(name: str, *, episodes: int, seed: int, cache: Path) -> tuple[dict, dict]:
    provenance, checkpoints = restore_run(name)
    final = load_model(checkpoints["final"])
    histories = {}
    for split, offset in (("fit", 0), ("test", 1)):
        histories[split] = collect_histories(
            final, provenance["recipe"]["environment"], episodes=episodes, seed=seed + offset,
        )
    features = {}
    for checkpoint, path in checkpoints.items():
        module = final if checkpoint == "final" else load_model(path)
        features[checkpoint] = {
            split: extract_layers(module, data.observations)
            for split, data in histories.items()
        }
    fit, test = histories["fit"], histories["test"]
    y_fit, y_test = fit.beliefs.reshape(-1, 6), test.beliefs.reshape(-1, 6)
    layers = SITES
    if features["final"]["fit"].shape[2] != len(layers):
        raise ValueError("expected four residual blocks and the final RMSNorm output")

    def feature_dict(checkpoint: str, split: str) -> dict:
        values = features[checkpoint][split]
        return {layer: values[:, :, i].reshape(-1, values.shape[-1]) for i, layer in enumerate(layers)}

    nuisance = {}
    for label, first, second in (
        ("current_token_action", fit.observations, test.observations),
        ("pending_token", fit.pending_tokens, test.pending_tokens),
        ("log_pending_token", np.log(np.maximum(fit.pending_tokens, 1e-12)), np.log(np.maximum(test.pending_tokens, 1e-12))),
    ):
        nuisance[label] = (first.reshape(-1, first.shape[-1]), second.reshape(-1, second.shape[-1]))
    init_fit, init_test = feature_dict("init", "fit"), feature_dict("init", "test")
    battery = evaluate_belief_geometry(
        feature_dict("final", "fit"), feature_dict("final", "test"),
        y_fit, y_test,
        train_groups=np.repeat(np.arange(episodes), HORIZON),
        test_groups=np.repeat(np.arange(episodes, 2 * episodes), HORIZON),
        nuisance_features=nuisance,
        initialization_features={layer: (init_fit[layer], init_test[layer]) for layer in layers},
        contrasts={
            "component_A_mass": np.array([1, 1, 1, 0, 0, 0]),
            "A_state0_minus_state1": np.array([1, -1, 0, 0, 0, 0]),
            "B_state0_minus_state1": np.array([0, 0, 0, 1, -1, 0]),
            "reward_state_mass": np.array([0, 0, 1, 0, 0, 1]),
        },
        seed=seed + 2, n_null_repeats=3, n_resamples=200,
    )
    predictions = {
        "init": {layer: battery.baseline_predictions[f"initialization/{layer}"] for layer in layers},
        "final": battery.predictions,
    }
    metrics = {
        checkpoint: {layer: geometry_metrics(value, y_test) for layer, value in values.items()}
        for checkpoint, values in predictions.items()
    }
    report = {
        "provenance": provenance,
        "protocol": {
            "episodes_per_split": episodes, "positions_per_episode": HORIZON,
            "fit_seed": seed, "test_seed": seed + 1, "analysis_seed": seed + 2,
            "sampling": "final checkpoint greedy policy; identical histories replayed through init and final",
            "target": "decision-time arrival belief conditioned on visible tokens and executed actions, without rewards",
            "timing": "t=0 is reset/BOS; t=1..127 reveal delayed tokens. t=127 is the terminal observation, with no action.",
            "warmup": 0,
            "representation": "post-final RMSNorm primary; residual after each of four blocks as robustness controls",
            "primary_layer": "post_final_norm (fixed before evaluation, not selected on test scores)",
            "probe": "affine; SVD cutoff chosen by whole-episode training-only cross-validation",
            "uncertainty": "fixed-probe episode bootstrap; one training seed per component set",
            "log_probability_floor": 1e-12,
            "cloud_selection": "seeded uniform sample of held-out rows, shared by both checkpoints",
            "example_selection": "first four held-out episodes from each true component; no selection on decoding",
            "coverage": {
                split: {
                    "component_episode_counts": np.bincount(data.components, minlength=2).tolist(),
                    "action_counts": np.bincount(data.actions[:, :-1].reshape(-1), minlength=3).tolist(),
                    "component_posterior_min": float(data.beliefs[:, :, :3].sum(axis=2).min()),
                    "component_posterior_max": float(data.beliefs[:, :, :3].sum(axis=2).max()),
                }
                for split, data in histories.items()
            },
            "interpretation": "linear accessibility on final-policy histories; not causal use or unique model identification",
        },
        "diagnostic_max_abs_error": max(fit.diagnostic_error, test.diagnostic_error),
        "metrics": metrics,
        "battery": battery.report,
    }
    cloud_rows = np.sort(np.random.default_rng(seed + 3).choice(len(y_test), min(6144, len(y_test)), replace=False))
    examples = np.sort(np.concatenate([np.flatnonzero(test.components == component)[:4] for component in range(2)]))
    viewer = build_simplex_run(
        name=name.replace("_", " ").title(),
        description=(
            f"{provenance['run_id']} · {provenance['agent_steps']:,} training steps · "
            + " / ".join(
                f"{label}: x={component['x']:.3f}, α={component['alpha']}"
                for label, component in zip(COMPONENT_INDICES, provenance["recipe"]["components"])
            )
        ),
        targets=test.beliefs,
        components=COMPONENT_INDICES,
        primary_site="post_final_norm",
        predictions={
            {"init": "Initialization", "final": "Final"}[checkpoint]: {
                layer: values.reshape(test.beliefs.shape) for layer, values in layer_values.items()
            }
            for checkpoint, layer_values in predictions.items()
        },
        cloud_rows=cloud_rows,
        example_episodes=examples,
        tokens=[["BOS" if t == 0 else str(token) for t, token in enumerate(row)] for row in test.tokens],
        episode_labels=[
            f"Episode {index} · true component {'AB'[component]}"
            for index, component in enumerate(test.components)
        ],
        position_notes=[
            [
                f"preceding executed action: {'none' if t == 0 else row[t - 1]} · "
                + ("terminal observation; no next action" if t == HORIZON - 1
                   else f"final-policy next action: {row[t]}")
                + " · action 0=noop, 1=positive, 2=negative"
                for t in range(HORIZON)
            ]
            for row in test.actions
        ],
    )
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache / f"{name}_heldout.npz", beliefs=test.beliefs, observations=test.observations,
        tokens=test.tokens, actions=test.actions, components=test.components,
        **{f"{checkpoint}_{layer}": value for checkpoint, values in predictions.items() for layer, value in values.items()},
    )
    return report, viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=160, help="independent episodes per fit/test split")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        for name in RUNS:
            provenance, paths = restore_run(name)
            for label, path in paths.items():
                model = load_model(path)
                print(name, label, provenance["agent_steps"], type(model).__name__, flush=True)
        return
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    viewer_runs = []
    reports = {}
    for index, name in enumerate(RUNS):
        print(f"Analyzing {name}", flush=True)
        report, viewer = analyze_run(
            name, episodes=args.episodes, seed=args.seed + 100 * index,
            cache=args.output / "raw",
        )
        reports[name] = report
        viewer_runs.append(viewer)
    analysis_metadata = {
        "source_sha256": digest(Path(__file__)),
        "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "library_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO.parent / "rl-harness", text=True).strip(),
        "numpy": np.__version__, "torch": torch.__version__,
    }
    summary = {
        "analysis": analysis_metadata,
        "runs": {name: {key: value for key, value in report.items() if key != "battery"} for name, report in reports.items()},
    }
    write_simplex_viewer(
        args.output, viewer_runs,
        title="Nonergodic belief geometry · Alpha 0.95",
        description=STUDY_DESCRIPTION,
        reports={**{f"{name} full report": report for name, report in reports.items()}, "Provenance and metrics": summary},
    )
    print(f"Viewer data and reports: {args.output}", flush=True)


if __name__ == "__main__":
    main()
