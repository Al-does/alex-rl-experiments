"""Run frozen-trunk next-token probes over every variant-1 checkpoint."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np
import torch

from devops.serverless.retrieve import (
    load_manifest,
    retrieve_manifest_artifacts,
)
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    ProbeData,
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.next_token_probe.probe import (
    ProbeTrainingConfig,
    build_sequence_dataset,
    fit_probe,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.shared import (
    BASE_MODEL_CONFIG,
    environment_config,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import (
    named_seed_sequences,
    seed_sequence_to_int,
)
from harness.storage.b2 import B2StorageConfig
from learners.models.transformer import TransformerModel


SOURCE_STUDY = "mess3_reward_state_action_asymmetry_cycle_5"
SOURCE_EXPERIMENT_REF = "1903c88dcb619f95394ee53de2df32b017f5de3a"
SOURCE_LIBRARY_REF = "2d2c2ff4cd57b5e10d08a18eaa76ffc4c4c73d2c"
SEEDS = (42, 43, 44, 45, 46)
CONTEXT_LENGTHS = (1, 10)
_STREAM_KEYS = {
    "probe_train": (700,),
    "probe_validation": (701,),
    "probe_test": (702,),
    "head_context_1": (703,),
    "head_context_10": (704,),
}
_MODULE_COMPONENTS = (
    "learner_group",
    "learner",
    "rl_module",
    "default_policy",
)


@dataclass(frozen=True, slots=True)
class SourceCheckpoint:
    name: str
    path: Path
    agent_steps: int
    training_iteration: int
    belief_metrics: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _LocalInferenceConfig:
    env: type
    env_config: dict[str, Any]
    num_env_runners: int = 0
    num_cpus_per_env_runner: int = 0
    num_gpus_per_env_runner: int = 0
    num_learners: int = 0
    num_gpus_per_learner: int = 0


@dataclass(frozen=True, slots=True)
class _LocalInferenceCheckpoint:
    module: Any
    config: _LocalInferenceConfig

    def get_module(self) -> Any:
        return self.module


def _module_checkpoint_path(checkpoint: Path) -> Path:
    """Return the default RLModule component within an Algorithm checkpoint."""

    module_checkpoint = checkpoint.joinpath(*_MODULE_COMPONENTS)
    state_files = (
        module_checkpoint / "module_state.pkl",
        module_checkpoint / "module_state.msgpack",
    )
    if (
        (module_checkpoint / "class_and_ctor_args.pkl").is_file()
        and any(path.is_file() for path in state_files)
    ):
        return module_checkpoint
    raise FileNotFoundError(
        "checkpoint has no restorable default RLModule component at "
        f"{module_checkpoint}"
    )


@contextmanager
def _load_local_inference_checkpoint(
    checkpoint: Path,
) -> Iterator[_LocalInferenceCheckpoint]:
    """Load only the checkpoint's local inference module, without an Algorithm."""

    env_config = environment_config(1)
    environment = HMMEnv(env_config)
    try:
        module = TransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            inference_only=True,
            model_config=dict(BASE_MODEL_CONFIG),
        )
    finally:
        environment.close()
    module.restore_from_path(
        _module_checkpoint_path(checkpoint),
        inference_only=True,
    )
    restored = _LocalInferenceCheckpoint(
        module=module,
        config=_LocalInferenceConfig(
            env=HMMEnv,
            env_config=env_config,
        ),
    )
    try:
        yield restored
    finally:
        del restored
        del module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _device(context: RunContext) -> str:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _source_prefix(config: B2StorageConfig, seed: int) -> str:
    run_id = f"mess3-rsa-c5-v1-seed{seed}"
    components = (
        config.prefix,
        "experiments",
        SOURCE_STUDY,
        "variant_1",
        run_id,
    )
    return "/".join(component.strip("/") for component in components if component)


def _download_source_artifacts(seed: int, destination: Path) -> dict[str, Any]:
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("cycle-5 checkpoint probing requires B2 credentials")
    prefix = _source_prefix(config, seed)
    manifest = load_manifest(
        key=f"{prefix}/metadata/durability_manifest.json",
        config=config,
    )
    retrieve_manifest_artifacts(
        manifest,
        destination,
        config=config,
    )
    return {
        "backend": "b2-s3",
        "base_uri": f"s3://{config.bucket}/{prefix}/",
        "manifest_key": f"{prefix}/metadata/durability_manifest.json",
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
    }


def _metric_value(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value in (None, ""):
        raise KeyError(f"progress.csv is missing {key!r}")
    return int(float(value))


def _source_checkpoints(root: Path) -> list[SourceCheckpoint]:
    progress_paths = list(root.rglob("progress.csv"))
    if len(progress_paths) != 1:
        raise RuntimeError(
            f"expected one source progress.csv, found {len(progress_paths)}"
        )
    with progress_paths[0].open(newline="") as handle:
        progress = list(csv.DictReader(handle))

    curve_paths = list(root.rglob("checkpoint_probe_curve.json"))
    if len(curve_paths) != 1:
        raise RuntimeError(
            "expected one source checkpoint_probe_curve.json, found "
            f"{len(curve_paths)}"
        )
    curve = json.loads(curve_paths[0].read_text())["checkpoints"]
    belief_by_name = {
        str(point.get("checkpoint_name", "initial_checkpoint")): point
        for point in curve
    }

    checkpoint_paths = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and (path / "algorithm_state.pkl").is_file()
        and (path / "class_and_ctor_args.pkl").is_file()
    ]
    by_name = {path.name: path for path in checkpoint_paths}
    expected_names = {
        "initial_checkpoint",
        *(f"checkpoint_{index:06d}" for index in range(len(progress))),
    }
    if set(by_name) != expected_names:
        missing = sorted(expected_names - set(by_name))
        extra = sorted(set(by_name) - expected_names)
        raise RuntimeError(
            f"source checkpoint set mismatch; missing={missing}, extra={extra}"
        )

    records = [
        SourceCheckpoint(
            name="initial_checkpoint",
            path=by_name["initial_checkpoint"],
            agent_steps=0,
            training_iteration=0,
            belief_metrics=belief_by_name.get("initial_checkpoint"),
        )
    ]
    step_key = "env_runners/num_env_steps_sampled_lifetime"
    for index, row in enumerate(progress):
        name = f"checkpoint_{index:06d}"
        records.append(
            SourceCheckpoint(
                name=name,
                path=by_name[name],
                agent_steps=_metric_value(row, step_key),
                training_iteration=_metric_value(row, "training_iteration"),
                belief_metrics=belief_by_name.get(name),
            )
        )
    return records


def _collect_checkpoint_data(
    checkpoint: Path,
    context: RunContext,
    streams: dict[str, np.random.SeedSequence],
) -> tuple[
    dict[str, ProbeData],
    np.ndarray,
    np.ndarray,
    float,
]:
    train_steps = 1_024 if context.smoke else 60_000
    validation_steps = 512 if context.smoke else 20_000
    test_steps = 1_024 if context.smoke else 80_000
    warmup = 4 if context.smoke else 64
    n_envs = 4 if context.smoke else 16
    with _load_local_inference_checkpoint(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        if int(env_config.get("delay", -1)) != 0:
            raise ValueError("cycle-5 next-token protocol requires delay=0")
        env_config["diagnostics"] = {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        }

        def make_environment():
            return env_class(env_config)

        environment = make_environment()
        try:
            transducer_target = make_transducer_target(environment)
            transition_matrices = np.stack(
                [
                    environment.task.transition_matrix_for_action(action)
                    for action in range(3)
                ]
            )
            emission_matrix = np.asarray(
                environment.model.emission_matrix,
                dtype=np.float64,
            ).copy()
        finally:
            environment.close()
        common = {
            "module": module,
            "env_factory": make_environment,
            "policy_mode": "greedy",
            "n_envs": n_envs,
            "device": _device(context),
            "warmup": warmup,
            "initial_belief": transducer_target[0],
            "action_outcome_operator": transducer_target[1],
            "initial_outcome_operator": transducer_target[2],
        }
        datasets = {
            "train": collect_probe_data(
                n_steps=train_steps,
                seed=streams["probe_train"],
                **common,
            ),
            "validation": collect_probe_data(
                n_steps=validation_steps,
                seed=streams["probe_validation"],
                **common,
            ),
            "test": collect_probe_data(
                n_steps=test_steps,
                seed=streams["probe_test"],
                **common,
            ),
        }

    target_error = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in datasets.values()
    )
    return datasets, transition_matrices, emission_matrix, target_error


def _belief_metrics(
    source: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if source is None:
        return None
    names = (
        "mse",
        "target_variance",
        "global_mse_ratio",
        "branch_baseline_mse",
        "fine_mse_ratio",
    )
    return {name: source[name] for name in names}


def run(context: RunContext) -> dict[str, Any]:
    """Probe all 23 checkpoints for one of the five variant-1 seeds."""

    if context.seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    training_config = (
        ProbeTrainingConfig(
            batch_size=256,
            max_epochs=4,
            patience=2,
        )
        if context.smoke
        else ProbeTrainingConfig()
    )
    protocol = {
        "source_study": SOURCE_STUDY,
        "source_experiment_ref": SOURCE_EXPERIMENT_REF,
        "source_library_ref": SOURCE_LIBRARY_REF,
        "variant": 1,
        "seed": context.seed,
        "checkpoint_selection": (
            "initial_only" if context.smoke else "initial_and_all_22_training"
        ),
        "policy_mode": "greedy_matching_affine_mse_probe",
        "representation": "post_final_layer_norm",
        "context_lengths": list(CONTEXT_LENGTHS),
        "action_conditions": ["blind", "selected_action_at_final_position"],
        "target": "exact_next_visible_token_distribution",
        "sampled_target_metric": "next_visible_token",
        "loss": (
            "soft_cross_entropy; equals forward KL plus fixed target entropy"
        ),
        "probe_training": asdict(training_config),
        "interpretation": {
            "context_1": "decodability from the current trunk state",
            "context_10": "recoverability from the recent trunk trajectory",
            "action_blind": "prediction before observing the selected action",
            "action_conditioned": (
                "prediction after policy selection and before environment step"
            ),
        },
    }
    outputs.write_json("probe_protocol.json", protocol)

    with tempfile.TemporaryDirectory(
        prefix=f"cycle5-next-token-seed{context.seed}-"
    ) as temporary:
        source_root = Path(temporary)
        source_artifacts = _download_source_artifacts(
            context.seed,
            source_root,
        )
        checkpoints = _source_checkpoints(source_root)
        if context.smoke:
            checkpoints = checkpoints[:1]

        points: list[dict[str, Any]] = []
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            rollout, transitions, emissions, target_error = (
                _collect_checkpoint_data(
                    checkpoint.path,
                    context,
                    streams,
                )
            )
            conditions: list[dict[str, Any]] = []
            for context_len in CONTEXT_LENGTHS:
                sequence_data = {
                    split: build_sequence_dataset(
                        data,
                        context_len=context_len,
                        transition_matrices=transitions,
                        emission_matrix=emissions,
                    )
                    for split, data in rollout.items()
                }
                head_seed = seed_sequence_to_int(
                    streams[f"head_context_{context_len}"],
                    bits=32,
                )
                for condition_on_action in (False, True):
                    conditions.append(
                        fit_probe(
                            sequence_data["train"],
                            sequence_data["validation"],
                            sequence_data["test"],
                            condition_on_action=condition_on_action,
                            device=_device(context),
                            seed=head_seed,
                            config=training_config,
                        )
                    )
            point = {
                "checkpoint_name": checkpoint.name,
                "checkpoint_index": checkpoint_index,
                "agent_steps": checkpoint.agent_steps,
                "training_iteration": checkpoint.training_iteration,
                "belief_probe": _belief_metrics(checkpoint.belief_metrics),
                "target_consistency_max_abs": target_error,
                "conditions": conditions,
            }
            points.append(point)
            _write_json(
                context.results_dir
                / "checkpoint_results"
                / f"{checkpoint.name}.json",
                point,
            )
            outputs.write_json(
                "next_token_probe_curve.json",
                {
                    "schema_version": 1,
                    "protocol": protocol,
                    "source_artifacts": source_artifacts,
                    "checkpoints": points,
                },
            )

    summary = {
        "schema_version": 1,
        "protocol": protocol,
        "source_artifacts": source_artifacts,
        "checkpoint_count": len(points),
        "condition_count_per_checkpoint": 4,
        "checkpoints": points,
    }
    outputs.write_json("next_token_probe_curve.json", summary)
    return summary
