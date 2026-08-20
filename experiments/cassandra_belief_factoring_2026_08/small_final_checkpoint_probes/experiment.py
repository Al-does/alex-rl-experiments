"""Probe only the final checkpoints of the two small trained models."""

from dataclasses import replace
from pathlib import Path

from experiments.cassandra_belief_factoring_2026_08.analysis import (
    probe_checkpoint,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext


AGENT_STEPS = 10_027_008
CHECKPOINT_SPECS = {
    "targeted": Path(
        "targeted_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m"
    )
    / "artifacts"
    / "20260820T060017Z-3d46dd5a"
    / "tune"
    / "PPO_CassandraActionObservationEnv_6ec2c_00000_0_2026-08-20_06-00-24"
    / "checkpoint_000000",
    "global_alias": Path(
        "global_alias_ppo_entropy_0_03_gamma_0_990_small_4layer_all_good_10m"
    )
    / "artifacts"
    / "20260820T081146Z-1d660cb4"
    / "tune"
    / "PPO_CassandraActionObservationEnv_cb68d_00000_0_2026-08-20_08-11-50"
    / "checkpoint_000000",
}


def checkpoint_paths(context: RunContext) -> dict[str, Path]:
    """Resolve source checkpoints relative to the Cassandra study root."""

    study_root = context.experiment_dir.parent
    return {
        name: study_root / relative
        for name, relative in CHECKPOINT_SPECS.items()
    }


def run(context: RunContext):
    """Run matched full probes on the two final checkpoints."""

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    results = {}
    paths = checkpoint_paths(context)
    for name, checkpoint in paths.items():
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"{name} final checkpoint not found: {checkpoint}"
            )
        result = probe_checkpoint(
            replace(
                context,
                results_dir=context.results_dir / name,
                resume_from=checkpoint,
            ),
            checkpoint=checkpoint,
            condition=f"small_{name}_final",
            agent_steps=AGENT_STEPS,
        )
        results[name] = {
            "checkpoint": checkpoint,
            "metrics": result.metrics,
        }

    outputs.write_json(
        "final_checkpoint_probe_comparison.json",
        {
            "agent_steps": AGENT_STEPS,
            "probe_scope": "final checkpoints only",
            "conditions": results,
        },
    )
    outputs.write_json(
        "output_validation.json",
        {
            "status": "completed",
            "required_files": [
                "targeted/probe_metrics.json",
                "global_alias/probe_metrics.json",
                "final_checkpoint_probe_comparison.json",
            ],
        },
    )
    return results


__all__ = ["AGENT_STEPS", "CHECKPOINT_SPECS", "checkpoint_paths", "run"]
