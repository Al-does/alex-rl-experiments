"""Build the passive three-checkpoint simplex viewer for the completed PPO run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from analysis.belief_geometry import (
    evaluate_belief_geometry,
    prediction_null_basis,
)
from analysis.checkpoints import load_module_only
from analysis.probes.controls import fit_grouped_affine, score_prediction
from analysis.probes.transducer import predictive_belief_sequence
from analysis.simplex import (
    ExplorerScore,
    build_simplex_run,
    geometry_metrics,
    write_nonergodic_belief_explorer,
)
from envs.hmm import HMMEnv
from harness.storage.b2 import B2StorageConfig

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
)
from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_COUNT,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    STATES_PER_COMPONENT,
    environment_config,
    nonergodic_mess3_model,
)


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RUN_ID = "20260912T043214Z-1bbe2a96"
RESULTS = ROOT / "ppo_h100_estimated" / "results" / RUN_ID
ARTIFACTS = ROOT / "ppo_h100_estimated" / "artifacts" / RUN_ID
SITE = "post_final_norm"
COMPONENTS = {"Component A": [0, 1, 2], "Component B": [3, 4, 5]}
CHECKPOINTS = {
    "Initialization": ("initial_checkpoint", 0),
    "2.16M steps": ("iteration_000008_steps_002162048", 2_162_048),
    "Final": ("checkpoint_000000", 15_134_336),
}
LOG_PROBABILITY_FLOOR = 1e-12
TOKEN_LABELS = list(nonergodic_mess3_model().token_labels)


@dataclass(frozen=True)
class Histories:
    observations: np.ndarray
    source_beliefs: np.ndarray
    pending_tokens: np.ndarray
    visible_tokens: np.ndarray
    components: np.ndarray
    diagnostic_error: float


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
    ).strip()


def restore_checkpoints() -> tuple[dict[str, object], dict[str, Path]]:
    manifest = json.loads((RESULTS / "run_manifest.json").read_text())
    recipe = json.loads((RESULTS / "resolved_recipe.json").read_text())
    trial = json.loads((RESULTS / "tune_summary.json").read_text())["trials"][0]
    archived = json.loads((RESULTS / "condition_summary.json").read_text())
    remote = manifest["remote_artifacts"]
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 credentials are required to restore checkpoints")
    client = config.s3_client()
    raw = client.get_object(
        Bucket=remote["bucket"],
        Key=remote["canonical_manifest_key"],
    )["Body"].read()
    durability = json.loads(raw)
    final_parts = Path(trial["checkpoint"]).parts
    final_relative = Path(*final_parts[final_parts.index(RUN_ID) + 1 :])
    module_suffix = Path("learner_group/learner/rl_module/default_policy")
    roots = {
        "Initialization": Path("initial_checkpoint") / module_suffix,
        "2.16M steps": (
            Path("log_spaced_checkpoints")
            / "iteration_000008_steps_002162048"
            / module_suffix
        ),
        "Final": final_relative / module_suffix,
    }
    local_root = ARTIFACTS / "simplex_checkpoints"
    paths: dict[str, Path] = {}
    checkpoint_provenance: dict[str, object] = {}
    for label, prefix in roots.items():
        files = [
            entry
            for entry in durability["files"]
            if Path(entry["relative_path"]).parent == prefix
        ]
        expected = {
            "class_and_ctor_args.pkl",
            "metadata.json",
            "module_state.pkl",
        }
        if {Path(entry["relative_path"]).name for entry in files} != expected:
            raise ValueError(f"unexpected module subtree for {label}")
        for entry in files:
            destination = local_root / entry["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                not destination.exists()
                or destination.stat().st_size != entry["size_bytes"]
                or digest(destination) != entry["sha256"]
            ):
                client.download_file(
                    remote["bucket"],
                    entry["key"],
                    str(destination),
                )
            if (
                destination.stat().st_size != entry["size_bytes"]
                or digest(destination) != entry["sha256"]
            ):
                raise ValueError(f"checkpoint checksum mismatch: {label}")
        paths[label] = local_root / prefix
        checkpoint_provenance[label] = {
            "checkpoint": CHECKPOINTS[label][0],
            "agent_steps": CHECKPOINTS[label][1],
            "module_path": str(paths[label].relative_to(REPO)),
            "files": {
                Path(entry["relative_path"]).name: {
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                }
                for entry in files
            },
        }
    archived_policy = {
        label: next(
            record["policy"]
            for record in archived["checkpoint_probes"]
            if record["checkpoint"] == checkpoint
        )
        for label, (checkpoint, _) in CHECKPOINTS.items()
    }
    return (
        {
            "run_id": RUN_ID,
            "training_git": manifest["git"],
            "recipe": recipe,
            "remote": {
                "endpoint": remote["endpoint"],
                "bucket": remote["bucket"],
                "artifact_prefix": remote["prefix"],
                "canonical_manifest_key": remote["canonical_manifest_key"],
                "durability_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "remote_file_count": remote["file_count"],
                "remote_total_size_bytes": remote["total_bytes"],
            },
            "checkpoints": checkpoint_provenance,
            "archived_policy_evaluation": archived_policy,
        },
        paths,
    )


def load_model(path: Path) -> FactoredReproductionActorCritic:
    model = load_module_only(path)
    if not isinstance(model, FactoredReproductionActorCritic):
        raise TypeError("expected a FactoredReproductionActorCritic checkpoint")
    return model.cpu().eval()


def _collect_episode(
    seed: int, guess: int
) -> tuple[np.ndarray, list[Mapping[str, object]]]:
    config = {
        **environment_config(),
        "diagnostics": {
            "belief": True,
            "state": True,
            "tokens": True,
            "transitions": True,
        },
    }
    env = HMMEnv(config)
    observations: list[np.ndarray] = []
    infos: list[Mapping[str, object]] = []
    try:
        observation, info = env.reset(seed=seed)
        for position in range(CONTEXT_LENGTH):
            observations.append(np.asarray(observation, dtype=np.float32))
            infos.append(info)
            if position == CONTEXT_LENGTH - 1:
                break
            observation, _, terminated, truncated, info = env.step(guess)
            if bool(terminated or truncated) != (position == EPISODE_LENGTH - 1):
                raise AssertionError("unexpected episode boundary")
    finally:
        env.close()
    return np.stack(observations), infos


def verify_passive_action_invariance(seed: int) -> None:
    observations_a, infos_a = _collect_episode(seed, 0)
    observations_b, infos_b = _collect_episode(seed, 2)
    if not np.array_equal(observations_a, observations_b):
        raise AssertionError("token histories changed with token guesses")
    for key in ("belief_current", "state_current", "visible_token_current"):
        values_a = np.asarray([info[key] for info in infos_a])
        values_b = np.asarray([info[key] for info in infos_b])
        if not np.array_equal(values_a, values_b):
            raise AssertionError(f"passive diagnostic {key} changed with guesses")


def collect_histories(*, episodes: int, seed: int) -> Histories:
    if episodes < 2:
        raise ValueError("each split needs at least two complete episodes")
    model = nonergodic_mess3_model()
    spawned = np.random.SeedSequence(seed).spawn(episodes)
    observations = np.empty(
        (episodes, CONTEXT_LENGTH, model.n_tokens),
        dtype=np.float32,
    )
    source_beliefs = np.empty(
        (episodes, CONTEXT_LENGTH, model.n_states),
        dtype=np.float64,
    )
    pending_tokens = np.empty(
        (episodes, CONTEXT_LENGTH, model.n_tokens),
        dtype=np.float64,
    )
    visible_tokens = np.full(
        (episodes, CONTEXT_LENGTH),
        -1,
        dtype=np.int64,
    )
    components = np.empty(episodes, dtype=np.int64)
    diagnostic_error = 0.0
    for episode, sequence in enumerate(spawned):
        episode_seed = int(sequence.generate_state(1)[0])
        episode_observations, infos = _collect_episode(episode_seed, 0)
        tokens = np.asarray(
            [int(info["visible_token_current"]) for info in infos[1:]],
            dtype=np.int64,
        )
        beliefs = predictive_belief_sequence(
            model.initial_distribution,
            model.edge_transition_matrices[tokens],
        )
        arrival = np.stack(
            [np.asarray(info["belief_current"], dtype=np.float64) for info in infos]
        )
        error = float(np.abs(beliefs @ model.transition_matrix - arrival).max())
        diagnostic_error = max(diagnostic_error, error)
        states = np.asarray(
            [int(info["state_current"]) for info in infos],
            dtype=np.int64,
        )
        component = states // STATES_PER_COMPONENT
        if not np.all(component == component[0]):
            raise AssertionError("an episode crossed nonergodic components")
        observations[episode] = episode_observations
        source_beliefs[episode] = beliefs
        pending_tokens[episode] = beliefs @ model.emission_matrix
        visible_tokens[episode, 1:] = tokens
        components[episode] = component[0]
        if (episode + 1) % 32 == 0 or episode + 1 == episodes:
            print(
                f"histories seed={seed}: {episode + 1}/{episodes}",
                flush=True,
            )
    if diagnostic_error > 1e-10:
        raise AssertionError(
            "source filter disagrees with public arrival-belief diagnostics: "
            f"{diagnostic_error}"
        )
    return Histories(
        observations=observations,
        source_beliefs=source_beliefs,
        pending_tokens=pending_tokens,
        visible_tokens=visible_tokens,
        components=components,
        diagnostic_error=diagnostic_error,
    )


@torch.inference_mode()
def extract_features(
    model: FactoredReproductionActorCritic,
    observations: np.ndarray,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    batches = []
    for start in range(0, len(observations), batch_size):
        tensor = torch.from_numpy(observations[start : start + batch_size])
        batches.append(model.encoder.forward_complete_episode(tensor).cpu().numpy())
    return np.concatenate(batches)


def _flat(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1, values.shape[-1])


def predictive_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
    fit: Histories,
    test: Histories,
    train_groups: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    report: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    targets = {
        "pending_token": (
            _flat(fit.pending_tokens),
            _flat(test.pending_tokens),
        ),
        "log_pending_token": (
            np.log(np.maximum(_flat(fit.pending_tokens), LOG_PROBABILITY_FLOOR)),
            np.log(np.maximum(_flat(test.pending_tokens), LOG_PROBABILITY_FLOOR)),
        ),
    }
    for name, (train_target, test_target) in targets.items():
        weight, bias, metadata = fit_grouped_affine(
            train_features,
            train_target,
            train_groups,
        )
        prediction = test_features @ weight + bias
        predictions[name] = prediction
        report[name] = {
            "fit": metadata,
            "metrics": score_prediction(prediction, test_target),
        }
    return report, predictions


def contrasts() -> dict[str, np.ndarray]:
    model = nonergodic_mess3_model()
    result = {
        "component_A_mass": np.asarray([1, 1, 1, 0, 0, 0]),
        "A_state0_minus_state1": np.asarray([1, -1, 0, 0, 0, 0]),
        "B_state0_minus_state1": np.asarray([0, 0, 0, 1, -1, 0]),
    }
    for index, vector in enumerate(
        prediction_null_basis(model.emission_matrix).T,
        start=1,
    ):
        result[f"pending_token_null_{index}"] = vector
    return result


def select_examples(components: np.ndarray) -> np.ndarray:
    return np.sort(
        np.concatenate(
            [
                np.flatnonzero(components == component)[:4]
                for component in range(COMPONENT_COUNT)
            ]
        )
    )


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def analyze(
    *,
    output: Path,
    summary_output: Path | None,
    episodes: int,
    seed: int,
) -> None:
    if output.exists():
        raise FileExistsError("output already exists; choose a new directory")
    provenance, checkpoint_paths = restore_checkpoints()
    verify_passive_action_invariance(seed + 10_000)
    histories = {
        "fit": collect_histories(episodes=episodes, seed=seed),
        "test": collect_histories(episodes=episodes, seed=seed + 1),
    }
    features: dict[str, dict[str, np.ndarray]] = {}
    for label, path in checkpoint_paths.items():
        model = load_model(path)
        features[label] = {
            split: extract_features(model, data.observations)
            for split, data in histories.items()
        }
        print(f"features {label}: complete", flush=True)

    fit = histories["fit"]
    test = histories["test"]
    train_groups = np.repeat(np.arange(episodes), CONTEXT_LENGTH)
    test_groups = np.repeat(
        np.arange(episodes, 2 * episodes),
        CONTEXT_LENGTH,
    )
    y_fit = _flat(fit.source_beliefs)
    y_test = _flat(test.source_beliefs)
    nuisance = {
        "visible_token": (
            _flat(fit.observations),
            _flat(test.observations),
        ),
        "pending_token": (
            _flat(fit.pending_tokens),
            _flat(test.pending_tokens),
        ),
        "log_pending_token": (
            np.log(np.maximum(_flat(fit.pending_tokens), LOG_PROBABILITY_FLOOR)),
            np.log(np.maximum(_flat(test.pending_tokens), LOG_PROBABILITY_FLOOR)),
        ),
    }
    batteries = {}
    belief_predictions: dict[str, np.ndarray] = {}
    prediction_reports = {}
    predictive_predictions: dict[str, dict[str, np.ndarray]] = {}
    init_pair = (
        _flat(features["Initialization"]["fit"]),
        _flat(features["Initialization"]["test"]),
    )
    for label in CHECKPOINTS:
        train_features = _flat(features[label]["fit"])
        test_features = _flat(features[label]["test"])
        initialization_features = (
            None if label == "Initialization" else {SITE: init_pair}
        )
        battery = evaluate_belief_geometry(
            {SITE: train_features},
            {SITE: test_features},
            y_fit,
            y_test,
            train_groups=train_groups,
            test_groups=test_groups,
            nuisance_features=nuisance,
            initialization_features=initialization_features,
            contrasts=contrasts(),
            seed=seed + 2,
            n_null_repeats=3,
            n_resamples=200,
        )
        batteries[label] = battery.report
        belief_predictions[label] = battery.predictions[SITE]
        prediction_report, prediction_values = predictive_scores(
            train_features,
            test_features,
            fit,
            test,
            train_groups,
        )
        prediction_reports[label] = prediction_report
        predictive_predictions[label] = prediction_values
        print(f"probes {label}: complete", flush=True)

    shapes = {value.shape for value in belief_predictions.values()}
    if shapes != {(episodes * CONTEXT_LENGTH, y_test.shape[1])}:
        raise AssertionError("checkpoint predictions are not row-aligned")
    metrics = {
        label: geometry_metrics(
            prediction,
            y_test,
            components=COMPONENTS,
        )
        for label, prediction in belief_predictions.items()
    }
    protocol = {
        "episodes_per_split": episodes,
        "positions_per_episode": CONTEXT_LENGTH,
        "fit_seed": seed,
        "test_seed": seed + 1,
        "analysis_seed": seed + 2,
        "split": (
            "independent complete episodes; each episode remains wholly in fit "
            "or test and no trajectory is shared across splits"
        ),
        "matched_histories": (
            "the same observations and source-belief targets are replayed "
            "through initialization, 2.16M-step, and final modules"
        ),
        "sampling": (
            "passive environment histories collected with a fixed token guess; "
            "verified identical under a different guess"
        ),
        "target": (
            "filtered source belief updated only by visible delayed tokens; "
            "actions and rewards are excluded"
        ),
        "pending_token_timing": (
            "pending-token probabilities equal source belief @ emission matrix"
        ),
        "diagnostic_alignment": (
            "source belief @ transition matrix equals public belief_current"
        ),
        "positions": (
            "position 0 is BOS/reset; positions 1-127 expose delayed visible "
            "tokens; position 127 remains visible with no next action"
        ),
        "warmup": 0,
        "representation_site": ("post_final_norm, selected before held-out evaluation"),
        "probe": (
            "raw affine predictions; SVD cutoff chosen by whole-episode "
            "training-only cross-validation"
        ),
        "controls": [
            "visible_token",
            "pending_token",
            "log_pending_token",
            "actual initialization checkpoint",
        ],
        "uncertainty": (
            "fixed-probe episode bootstrap with 200 resamples; three null repeats"
        ),
        "log_probability_floor": LOG_PROBABILITY_FLOOR,
        "cloud_selection": (
            "seeded uniform sample of held-out rows, shared by all checkpoints"
        ),
        "example_selection": (
            "first four held-out episodes from each true component; no "
            "selection on decoding scores"
        ),
        "interpretation": (
            "descriptive linear accessibility, not causal use or unique model "
            "identification; one training seed limits across-model generality"
        ),
        "coverage": {
            split: {
                "component_episode_counts": np.bincount(
                    data.components,
                    minlength=COMPONENT_COUNT,
                ).tolist(),
                "source_component_A_min": float(
                    data.source_beliefs[:, :, :STATES_PER_COMPONENT].sum(axis=2).min()
                ),
                "source_component_A_max": float(
                    data.source_beliefs[:, :, :STATES_PER_COMPONENT].sum(axis=2).max()
                ),
            }
            for split, data in histories.items()
        },
    }
    analysis_metadata = {
        "source_sha256": digest(Path(__file__)),
        "experiment_commit": git_revision(REPO),
        "library_commit": git_revision(REPO.parent / "rl-harness"),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    report = {
        "analysis": analysis_metadata,
        "provenance": provenance,
        "protocol": protocol,
        "diagnostic_max_abs_error": max(
            fit.diagnostic_error,
            test.diagnostic_error,
        ),
        "metrics": metrics,
        "predictive_scores": prediction_reports,
        "battery": batteries,
    }
    compact = {
        "analysis": analysis_metadata,
        "run_id": RUN_ID,
        "checkpoints": {
            label: {
                "agent_steps": CHECKPOINTS[label][1],
                "checkpoint_files": provenance["checkpoints"][label]["files"],
                "geometry": metrics[label],
                "activation_to_pending_token": prediction_reports[label][
                    "pending_token"
                ]["metrics"],
                "activation_to_log_pending_token": prediction_reports[label][
                    "log_pending_token"
                ]["metrics"],
                "archived_policy_evaluation": provenance["archived_policy_evaluation"][
                    label
                ],
            }
            for label in CHECKPOINTS
        },
        "protocol": protocol,
        "diagnostic_max_abs_error": report["diagnostic_max_abs_error"],
    }

    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    np.savez_compressed(
        raw / "heldout_affine_predictions.npz",
        source_beliefs=test.source_beliefs,
        pending_token_probabilities=test.pending_tokens,
        observations=test.observations,
        visible_tokens=test.visible_tokens,
        components=test.components,
        **{
            f"{label.lower().replace(' ', '_').replace('.', '')}_belief_prediction": (
                value.reshape(test.source_beliefs.shape)
            )
            for label, value in belief_predictions.items()
        },
        **{
            (
                f"{label.lower().replace(' ', '_').replace('.', '')}_"
                f"{target}_prediction"
            ): prediction
            for label, targets in predictive_predictions.items()
            for target, prediction in targets.items()
        },
    )
    (raw / "provenance.json").write_text(
        json.dumps(json_safe(provenance), indent=2) + "\n"
    )
    (output / "analysis_report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n"
    )
    (output / "analysis_summary.json").write_text(
        json.dumps(json_safe(compact), indent=2) + "\n"
    )
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        if summary_output.exists():
            raise FileExistsError("summary output already exists; choose a new path")
        summary_output.write_text(json.dumps(json_safe(compact), indent=2) + "\n")

    flat_count = episodes * CONTEXT_LENGTH
    cloud_rows = np.sort(
        np.random.default_rng(seed + 3).choice(
            flat_count,
            min(6144, flat_count),
            replace=False,
        )
    )
    example_episodes = select_examples(test.components)
    token_labels = [
        [
            "BOS" if position == 0 else TOKEN_LABELS[token]
            for position, token in enumerate(row)
        ]
        for row in test.visible_tokens
    ]
    layer_scores = {
        label: {
            SITE: {
                "Activation → NTP R²": ExplorerScore(
                    prediction_reports[label]["pending_token"]["metrics"]["r_squared"],
                    (
                        "Independent complete held-out episodes; source-belief "
                        "pending-token timing; grouped training-only affine fit."
                    ),
                ),
                "Activation → log NTP R²": ExplorerScore(
                    prediction_reports[label]["log_pending_token"]["metrics"][
                        "r_squared"
                    ],
                    (
                        "Natural log with floor 1e-12; independent complete "
                        "held-out episodes; grouped training-only affine fit."
                    ),
                ),
            }
        }
        for label in CHECKPOINTS
    }
    viewer = build_simplex_run(
        name="Passive MESS3 PPO · seed 42",
        description=(
            f"Run {RUN_ID}. Actual initialization, 2,162,048-step, and "
            "15,134,336-step modules on matched passive histories. Source "
            "beliefs use visible delayed tokens only; token guesses and rewards "
            "do not enter the filter."
        ),
        targets=test.source_beliefs,
        predictions={
            label: {SITE: prediction.reshape(test.source_beliefs.shape)}
            for label, prediction in belief_predictions.items()
        },
        components=COMPONENTS,
        primary_site=SITE,
        cloud_rows=cloud_rows,
        example_episodes=example_episodes,
        tokens=token_labels,
        state_labels=list(nonergodic_mess3_model().state_labels),
        episode_labels=[
            f"Episode {index} · true component {'AB'[component]}"
            for index, component in enumerate(test.components)
        ],
        position_notes=[
            [
                (
                    "reset prior; no visible token"
                    if position == 0
                    else (
                        "visible delayed token updates source belief; "
                        + (
                            "terminal position; no next action"
                            if position == CONTEXT_LENGTH - 1
                            else "token guess does not affect transition"
                        )
                    )
                )
                for position in range(CONTEXT_LENGTH)
            ]
            for _ in range(episodes)
        ],
        layer_scores=layer_scores,
    )
    write_nonergodic_belief_explorer(
        output,
        [viewer],
        title="Passive Nonergodic MESS3 · Three PPO Checkpoints",
        description=(
            "Held-out source-belief geometry for the actual initialization, "
            "2.16M-step, and final checkpoints. Raw affine coordinates are "
            "shown without clipping or simplex projection."
        ),
        reports={
            "Full grouped-probe report": json_safe(report),
            "Compact summary and provenance": json_safe(compact),
        },
    )
    print(f"viewer and raw predictions: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--episodes",
        type=int,
        default=160,
        help="independent complete episodes per fit/test split",
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.download_only:
        provenance, paths = restore_checkpoints()
        for label, path in paths.items():
            model = load_model(path)
            print(
                label,
                provenance["checkpoints"][label]["agent_steps"],
                path,
                type(model).__name__,
            )
        return
    analyze(
        output=args.output,
        summary_output=args.summary_output,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
