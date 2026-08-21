"""Probe only the 50M final checkpoints of the two small trained models."""

from dataclasses import replace
from pathlib import Path

from experiments.cassandra_belief_factoring_2026_08.analysis import (
    probe_checkpoint,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext


AGENT_STEPS = 50_069_504
CHECKPOINT_SPECS = {
    "targeted": Path("targeted_ppo_small_continue_30m")
    / "artifacts"
    / "20260820T205945Z-e2f72153"
    / "checkpoints"
    / "iteration_000916_final",
    "global_alias": Path("global_alias_ppo_small_continue_30m")
    / "artifacts"
    / "20260820T204934Z-79b17e50"
    / "checkpoints"
    / "iteration_000916_final",
}


def checkpoint_paths(context: RunContext) -> dict[str, Path]:
    """Resolve source checkpoints relative to the Cassandra study root."""

    study_root = context.experiment_dir.parent
    return {
        name: study_root / relative
        for name, relative in CHECKPOINT_SPECS.items()
    }


def run(context: RunContext):
    """Run matched full probes on the two 50M final checkpoints."""

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    results = {}
    paths = checkpoint_paths(context)
    for name, checkpoint in paths.items():
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"{name} 50M final checkpoint not found: {checkpoint}"
            )
        result = probe_checkpoint(
            replace(
                context,
                results_dir=context.results_dir / name,
                resume_from=checkpoint,
            ),
            checkpoint=checkpoint,
            condition=f"small_{name}_50m_final",
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
            "probe_scope": "50M final checkpoints only",
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
