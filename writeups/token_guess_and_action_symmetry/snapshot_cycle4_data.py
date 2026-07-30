"""Create the compact cycle-4 trajectory snapshot used by this write-up."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "data" / "action_symmetry_cycle4_mse_curves.json"
RUN_PATTERN = re.compile(r"mess3-rsa-c4-v[123]-seed(\d+)$")


def snapshot(source_root: Path) -> dict[str, Any]:
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_files: list[str] = []
    for path in sorted(
        source_root.glob("variant_*/results/*/checkpoint_probe_curve.json")
    ):
        variant = path.parents[2].name
        match = RUN_PATTERN.fullmatch(path.parent.name)
        if match is None:
            raise ValueError(f"unexpected cycle-4 run directory: {path.parent}")
        seed = match.group(1)
        payload = json.loads(path.read_text())
        curves.setdefault(variant, {})[seed] = [
            {
                "checkpoint_index": index,
                "agent_steps": int(point["agent_steps"]),
                "training_iteration": point.get("training_iteration"),
                "mse": float(point["mse"]),
                "reward_state_2_fraction_greedy": float(
                    point["reward_state_2_fraction_greedy"]
                ),
                "greedy_action_fractions": point["greedy_action_fractions"],
            }
            for index, point in enumerate(payload["checkpoints"])
        ]
        source_files.append(str(path.relative_to(source_root)))

    expected_variants = {"variant_1", "variant_2", "variant_3"}
    if set(curves) != expected_variants:
        raise ValueError(
            f"expected {sorted(expected_variants)}, found {sorted(curves)}"
        )
    for variant, seeds in curves.items():
        if set(seeds) != {"42", "43", "44", "45", "46"}:
            raise ValueError(f"{variant} has unexpected seeds: {sorted(seeds)}")

    return {
        "study": "mess3_reward_state_action_symmetry_cycle_4",
        "source_experiment_ref": "8cd58c4f313c812d318863144e26cdad93d80fe2",
        "source_library_ref": "38f5136f3085743e5b3ba2e89d7e7fc247ba5b5e",
        "source_results_commit": "3d0b709",
        "metric": "held_out_affine_probe_mse",
        "representation": "post_final_layer_norm",
        "sampling_distribution": "process_weighted_rollout",
        "source_files": source_files,
        "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_root",
        type=Path,
        help="Extracted mess3_reward_state_action_symmetry_cycle_4 directory.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = snapshot(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
