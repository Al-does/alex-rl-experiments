"""Continue the interrupted Shot A variant 2 entropy-ramp run from 45.06M.

The first entropy-ramp continuation was stopped at ~140M lifetime steps; this
leaf restores its 45.06M-step checkpoint into a freshly built Shot A variant 2
config (the restored lifetime step count keeps the learning-rate schedule on
its terminal 1e-5 segment) with one override: the entropy coefficient ramps
linearly from 0.01 to 0.05 across 45.06M-55.06M lifetime steps and holds 0.05
afterwards. Training runs to the same 315.06M lifetime target, saving a public
checkpoint at every 15M-step boundary and probing each saved checkpoint with
the study's standard battery.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from typing import Any

from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.shared import (
    SMOKE_ENV_STEPS,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.shot_a.shared import (
    build_config as build_shot_a_config,
    resolved_recipe as resolved_shot_a_recipe,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_algorithm
from harness.storage.b2 import B2StorageConfig


VARIANT = 2
CONDITION = "shot_a_variant_2_entropy_ramp_2"
SOURCE_RUN_ID = "20260920T073522Z-bed2a40f"
SOURCE_RESULT_COMMIT = "8297e6d589a932f65fdf4b11b52f2b1e5fdf3e75"
SOURCE_B2_PREFIX = (
    "experiments/nonergodic_mess3_reward_state_action_symmetry_cycle_5/"
    f"shot_a/variant_2_entropy_ramp/{SOURCE_RUN_ID}"
)
SOURCE_CHECKPOINT_KEY = (
    f"{SOURCE_B2_PREFIX}/checkpoints/iteration_000171_steps_045055536"
)
SOURCE_AGENT_STEPS = 45_055_536
SOURCE_TRAINING_ITERATION = 171
TOTAL_ENV_STEPS = 315_057_120
CHECKPOINT_EVERY_STEPS = 15_000_000
ENTROPY_COEFF_SCHEDULE = [
    [0, 0.0],
    [15_000_000, 0.0],
    [20_000_000, 0.01],
    [45_055_536, 0.01],
    [55_055_536, 0.05],
]
SMOKE_TOTAL_ENV_STEPS = 2 * SMOKE_ENV_STEPS
SMOKE_CHECKPOINT_EVERY_STEPS = SMOKE_ENV_STEPS
STEPS_METRIC = "env_runners/num_env_steps_sampled_lifetime"
CHECKPOINT_INDEX = "checkpoint_index.json"


def _lifetime_steps(result: Mapping[str, Any]) -> int:
    env_runners = result.get("env_runners")
    if isinstance(env_runners, Mapping):
        value = env_runners.get("num_env_steps_sampled_lifetime")
    else:
        value = result.get(STEPS_METRIC)
    if value is None:
        raise KeyError(STEPS_METRIC)
    return int(value)


def _training_iteration(result: Mapping[str, Any]) -> int:
    return int(result["training_iteration"])


def _write_index(index_path: Path, records: list[dict[str, Any]]) -> None:
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"checkpoints": records}, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(index_path)


def _save_step_boundary_checkpoint(
    *,
    algorithm: Any,
    result: Mapping[str, Any],
    checkpoint_root: str,
    every_steps: int,
    start_steps: int,
    **_: Any,
) -> None:
    """Save a public checkpoint the first time each step boundary is crossed."""

    steps = _lifetime_steps(result)
    iteration = _training_iteration(result)
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / CHECKPOINT_INDEX
    records: list[dict[str, Any]] = []
    if index_path.is_file():
        records = list(json.loads(index_path.read_text())["checkpoints"])
    last_saved = max(
        (int(record["agent_steps"]) for record in records),
        default=start_steps,
    )
    if steps // every_steps <= last_saved // every_steps:
        return
    destination = root / f"iteration_{iteration:06d}_steps_{steps:09d}"
    saved = Path(algorithm.save_to_path(str(destination)))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": iteration,
            "agent_steps": steps,
        }
    )
    _write_index(index_path, records)


def source_checkpoint_dir(context: RunContext) -> Path:
    return context.artifacts_dir / "source_checkpoint"


def fetch_source_checkpoint(context: RunContext) -> Path:
    """Download the source run's final Algorithm checkpoint from B2."""

    destination = source_checkpoint_dir(context)
    if (destination / "rllib_checkpoint.json").is_file():
        return destination
    storage = B2StorageConfig.from_env()
    if storage is None:
        raise RuntimeError(
            "continuation needs --resume-from or configured B2_* settings "
            "to fetch the source checkpoint"
        )
    prefix = f"{SOURCE_CHECKPOINT_KEY}/"
    if storage.prefix:
        prefix = f"{storage.prefix}/{prefix}"
    client = storage.s3_client()
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=storage.bucket, Prefix=prefix
    ):
        for entry in page.get("Contents", []):
            relative = entry["Key"][len(prefix) :]
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(storage.bucket, entry["Key"], str(target))
            count += 1
    if not (destination / "rllib_checkpoint.json").is_file():
        raise RuntimeError(
            f"source checkpoint missing after downloading {count} objects "
            f"from s3://{storage.bucket}/{prefix}"
        )
    return destination


def _is_checkpoint(path: Path) -> bool:
    return (path / "rllib_checkpoint.json").is_file()


def _restore_source(algorithm: Any, source: Path, rehomed: Path) -> None:
    """Load the source state and save it under this box's resource layout.

    The source checkpoint stores the config of the box that trained it, so
    restoring it verbatim can demand more env-runner CPUs than this machine
    has. Probes therefore read the re-saved copy, whose stored config fits.
    """

    algorithm.restore_from_path(str(source))
    if not _is_checkpoint(rehomed):
        algorithm.save_to_path(str(rehomed))


def rehome_source_checkpoint(config: Any, source: Path, rehomed: Path) -> Path:
    if not _is_checkpoint(rehomed):
        algorithm = config.build_algo()
        try:
            _restore_source(algorithm, source, rehomed)
        finally:
            algorithm.stop()
    return rehomed


class _RestoringConfig:
    """Build the fresh Shot A Algorithm, then load the source checkpoint."""

    def __init__(self, config: Any, checkpoint: Path, rehomed: Path) -> None:
        self._config = config
        self._checkpoint = checkpoint
        self._rehomed = rehomed

    def build_algo(self) -> Any:
        algorithm = self._config.build_algo()
        _restore_source(algorithm, self._checkpoint, self._rehomed)
        return algorithm

    def to_dict(self) -> dict[str, Any]:
        return self._config.to_dict()


def build_config(context: RunContext, checkpoint_root: Path):
    smoke = context.smoke
    return build_shot_a_config(context, VARIANT).training(
        entropy_coeff=ENTROPY_COEFF_SCHEDULE,
    ).callbacks(
        on_algorithm_init=None,
        on_train_result=partial(
            _save_step_boundary_checkpoint,
            checkpoint_root=str(checkpoint_root),
            every_steps=(
                SMOKE_CHECKPOINT_EVERY_STEPS if smoke else CHECKPOINT_EVERY_STEPS
            ),
            start_steps=0 if smoke else SOURCE_AGENT_STEPS,
        ),
    )


def _saved_checkpoints(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.is_file():
        return []
    return list(json.loads(index_path.read_text())["checkpoints"])


def _completed_training_result(
    outputs: RunArtifacts, index_path: Path, total_steps: int
) -> Mapping[str, Any] | None:
    """Return the recorded final iteration when this run already trained.

    Lets a rerun with the same ``--run-id`` (after a post-training failure)
    skip straight to probing instead of retraining from the source.
    """

    saved = _saved_checkpoints(index_path)
    if not saved or int(saved[-1]["agent_steps"]) < total_steps:
        return None
    if not _is_checkpoint(Path(saved[-1]["path"])):
        return None
    lines = [
        line
        for line in outputs.metrics_path.read_text().splitlines()
        if line.strip()
    ]
    final = json.loads(lines[-1])
    if _lifetime_steps(final) < total_steps:
        return None
    return final


def resolved_recipe(context: RunContext) -> dict[str, object]:
    recipe = resolved_shot_a_recipe(context, VARIANT)
    recipe.update(
        {
            "condition": CONDITION,
            "continuation_of": {
                "condition": "shot_a_variant_2_entropy_ramp",
                "run_id": SOURCE_RUN_ID,
                "result_commit": SOURCE_RESULT_COMMIT,
                "checkpoint_key": SOURCE_CHECKPOINT_KEY,
                "agent_steps": SOURCE_AGENT_STEPS,
                "training_iteration": SOURCE_TRAINING_ITERATION,
            },
            "restore": (
                "fresh Shot A variant 2 config built on this box, then "
                "Algorithm.restore_from_path of the source final checkpoint "
                "(learner, optimizer, connector, metrics, and iteration state)"
            ),
            "entropy_coeff": ENTROPY_COEFF_SCHEDULE,
            "entropy_schedule_note": (
                "restored lifetime step count resumes at 45.06M, so the "
                "coefficient ramps 0.01 -> 0.05 across 45.06M-55.06M "
                "lifetime steps and holds 0.05 for the remainder of training"
            ),
            "total_env_steps": (
                SMOKE_TOTAL_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
            "additional_env_steps": (
                (SMOKE_TOTAL_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS)
                - (0 if context.smoke else SOURCE_AGENT_STEPS)
            ),
            "note": (
                "source run 20260920T073522Z-bed2a40f was interrupted at "
                "~140M lifetime steps; this leaf resumes from its "
                "45.06M-step boundary checkpoint with a higher entropy cap"
            ),
            "stopping_metric": STEPS_METRIC,
            "checkpoint_schedule": (
                "source final, first iteration crossing each "
                f"{CHECKPOINT_EVERY_STEPS:,}-step boundary, final"
            ),
        }
    )
    return recipe


def run(context: RunContext) -> dict[str, Any]:
    from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.analysis import (
        analyze_checkpoint,
    )

    if context.seed is None:
        raise ValueError("non-ergodic action symmetry requires a resolved seed")
    if context.resume_from is not None:
        source = Path(context.resume_from)
        if not (source / "rllib_checkpoint.json").is_file():
            raise ValueError(f"{source} is not an Algorithm checkpoint")
    elif context.smoke:
        raise ValueError("smoke continuation requires --resume-from")
    else:
        source = fetch_source_checkpoint(context)

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", resolved_recipe(context))
    checkpoint_root = context.artifacts_dir / "step_checkpoints"
    rehomed_source = checkpoint_root / "source_final_checkpoint"
    index_path = checkpoint_root / CHECKPOINT_INDEX
    total_steps = SMOKE_TOTAL_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    final_result = _completed_training_result(outputs, index_path, total_steps)
    if final_result is None:
        final_result = run_algorithm(
            _RestoringConfig(
                build_config(context, checkpoint_root), source, rehomed_source
            ),
            replace(context, resume_from=None),
            should_stop=lambda result: _lifetime_steps(result) >= total_steps,
            checkpoint_at_end=True,
        )
    else:
        rehome_source_checkpoint(
            build_config(context, checkpoint_root), source, rehomed_source
        )
    write_training_curves(context)

    saved = _saved_checkpoints(index_path)
    final_steps = _lifetime_steps(final_result)
    if not saved or int(saved[-1]["agent_steps"]) < final_steps:
        for final_dir in sorted(outputs.checkpoints_dir.glob("*_final")):
            if not (final_dir / "rllib_checkpoint.json").is_file():
                continue
            saved.append(
                {
                    "path": str(final_dir),
                    "checkpoint_name": final_dir.name,
                    "training_iteration": _training_iteration(final_result),
                    "agent_steps": final_steps,
                }
            )
            break
    records: list[Mapping[str, Any]] = [
        {
            "path": str(rehomed_source),
            "checkpoint_name": rehomed_source.name,
            "training_iteration": (
                0 if context.smoke else SOURCE_TRAINING_ITERATION
            ),
            "agent_steps": 0 if context.smoke else SOURCE_AGENT_STEPS,
        },
        *saved,
    ]
    reports = []
    for record in records:
        reports.append(
            analyze_checkpoint(
                replace(
                    context,
                    results_dir=(
                        context.results_dir
                        / "checkpoint_probes"
                        / f"steps_{int(record['agent_steps']):09d}"
                    ),
                    resume_from=Path(record["path"]),
                ),
                variant=VARIANT,
                checkpoint=Path(record["path"]),
                checkpoint_label=str(record["checkpoint_name"]),
                agent_steps=int(record["agent_steps"]),
                training_iteration=int(record["training_iteration"]),
            )
        )
    summary = {
        "condition": CONDITION,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "continuation_of": resolved_recipe(context)["continuation_of"],
        "final_training_iteration": _training_iteration(final_result),
        "final_agent_steps": final_steps,
        "checkpoint_probes": [
            {
                "checkpoint": report["checkpoint"],
                "agent_steps": report["agent_steps"],
                "training_iteration": report["training_iteration"],
                "probe_fits": report["probe_fits"],
                "controls": report["controls"],
                "policy": report["policy"],
            }
            for report in reports
        ],
        "initial_probe": reports[0],
        "final_probe": reports[-1],
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
