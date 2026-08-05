"""Multi-checkpoint belief-symmetry probes for campaign 0040."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.analysis import (
    probe_checkpoint,
)
from harness.context import RunContext

# Training iterations to probe (init is handled separately).
PROBE_ITERATIONS = (2, 8, 22)
CHECKPOINT_LABELS = ("initial", "iter_2", "iter_8", "iter_22")


def _checkpoint_name_for_iteration(training_iteration: int) -> str:
    """Map Ray training_iteration to the saved checkpoint directory name."""
    if training_iteration < 1:
        raise ValueError("training_iteration must be positive")
    return f"checkpoint_{training_iteration - 1:06d}"


def _bundle_member_paths(bundle: Path) -> dict[str, Path]:
    """Resolve checkpoint directories inside a recovered source bundle."""
    members = {"initial": bundle / "initial_checkpoint"}
    for iteration in PROBE_ITERATIONS:
        label = f"iter_{iteration}"
        members[label] = bundle / label
    missing = [
        label
        for label, path in members.items()
        if not path.is_dir() or not any(path.rglob("*"))
    ]
    if missing:
        raise FileNotFoundError(
            f"missing checkpoint bundle members: {', '.join(missing)}"
        )
    return members


def _source_provenance(bundle: Path) -> dict[str, Any]:
    path = bundle / "source_provenance.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"source_run_id": bundle.name, "bundle": str(bundle.resolve())}


def run_probe_condition(context: RunContext, *, cycle: int, variant: int) -> dict[str, Any]:
    """Probe init plus training iterations 2, 8, and 22."""
    bundle = context.resume_from
    if bundle is None:
        raise ValueError(
            "resume_from must name a bundle with initial_checkpoint/ and iter_* members"
        )
    bundle = Path(bundle)
    checkpoints = _bundle_member_paths(bundle)
    context.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "study": "belief_symmetry_probes_0040",
        "cycle": cycle,
        "variant": variant,
        "seed": context.seed,
        "source": _source_provenance(bundle),
        "probe_iterations": list(PROBE_ITERATIONS),
        "checkpoint_labels": list(CHECKPOINT_LABELS),
        "target_definitions": {
            "symmetric_b2": "b2",
            "antisymmetric_b0_minus_b1": "b0-b1",
            **({"coarse_b2": "separate A={0,1}, B={2} lumped Bayes filter"} if variant in (1, 2) else {}),
        },
        "filter_definitions": {
            "full": "delay-0 initial measurement; later transition@measurement using action-dependent transitions",
            **({"coarse": "tokens 0/1 coarsened to not-2; destination-lump rows, never summed source rows"} if variant in (1, 2) else {}),
        },
        "random_weight_baseline_interpretation": (
            "The restored initial checkpoint estimates the affine random-network floor; "
            "it is a baseline, not evidence that an untrained network computes belief."
        ),
        "checkpoints": {},
    }
    for label, checkpoint in checkpoints.items():
        summary["checkpoints"][label] = probe_checkpoint(
            replace(context, resume_from=checkpoint),
            checkpoint,
            cycle=cycle,
            variant=variant,
            label=label,
        )
    (context.results_dir / "condition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
