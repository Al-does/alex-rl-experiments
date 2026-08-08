"""Probe untrained transformer checkpoints across architecture ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mess3_untrained_transformer_probe_ablation_2026_08.shared import (
    ARCHITECTURE_SPECS,
    DEFAULT_SEEDS,
    DEFAULT_TASK_VARIANT,
    ArchitectureSpec,
    run_ablation,
)
from harness.context import RunContext


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _select_specs(raw: str) -> tuple[ArchitectureSpec, ...]:
    if raw == "all":
        return ARCHITECTURE_SPECS
    wanted = {part.strip() for part in raw.split(",")}
    specs = tuple(spec for spec in ARCHITECTURE_SPECS if spec.key in wanted)
    missing = wanted - {spec.key for spec in specs}
    if missing:
        raise ValueError(f"unknown architecture keys: {sorted(missing)}")
    return specs


def run(context: RunContext) -> dict:
    return run_ablation(
        context,
        task_variant=DEFAULT_TASK_VARIANT,
        seeds=DEFAULT_SEEDS,
        specs=ARCHITECTURE_SPECS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-variant",
        type=int,
        default=DEFAULT_TASK_VARIANT,
        help="Fixed sticky-state HMM reward variant used for rollouts.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--architectures",
        default="all",
        help="Comma-separated architecture keys or 'all'.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    args = parser.parse_args()

    context = RunContext(
        experiment_dir=Path(__file__).resolve().parents[1],
        results_dir=args.results_dir.resolve(),
        artifacts_dir=args.artifacts_dir.resolve(),
        seed=DEFAULT_SEEDS[0],
        smoke=args.smoke,
    )
    payload = run_ablation(
        context,
        task_variant=args.task_variant,
        seeds=_parse_int_list(args.seeds),
        specs=_select_specs(args.architectures),
    )
    print(payload["summary"])


if __name__ == "__main__":
    main()
