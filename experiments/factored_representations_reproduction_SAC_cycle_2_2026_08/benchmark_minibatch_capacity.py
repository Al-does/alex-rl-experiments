"""Benchmark SAC learner batches split into exactly eight minibatches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME

from experiments.factored_representations_reproduction_SAC_cycle_2_2026_08.shared import (
    build_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES

MINIBATCH_COUNT = 8
DEFAULT_CANDIDATES = (
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
)


def _context(root: Path) -> RunContext:
    return RunContext(
        experiment_dir=Path(__file__).parent,
        results_dir=root / "results",
        artifacts_dir=root / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090_gpuinfer"],
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_single(*, batch_size: int, output: Path) -> int:
    """Run two updates for one candidate in an OOM-isolated process."""

    if batch_size % MINIBATCH_COUNT:
        raise ValueError(f"batch size must be divisible by {MINIBATCH_COUNT}")
    minibatch_size = batch_size // MINIBATCH_COUNT
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "minibatch_count": MINIBATCH_COUNT,
        "minibatch_size": minibatch_size,
        "factor_count": 3,
        "condition": "sac_aux_ce",
        "status": "failed",
    }
    algorithm = None
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        context = _context(output.parent / "runtime")
        config = build_config(
            context,
            factor_count=3,
            condition="sac_aux_ce",
            target_entropy_fraction=0.6,
            auxiliary_coefficient=0.3,
        )
        config.callbacks(on_algorithm_init=None, on_train_result=None)
        config.training(num_steps_sampled_before_learning_starts=0)
        algorithm = config.build_algo()
        # Seed the replay buffer with real environment episodes. Candidate
        # batches are then sampled with replacement, as in ordinary SAC.
        algorithm.train()
        sample_started = time.perf_counter()
        episodes = algorithm.local_replay_buffer.sample(
            num_items=batch_size,
            n_step=config.n_step,
            batch_length_T=0,
            lookback=0,
            min_batch_length_T=0,
            gamma=config.gamma,
            beta=config.replay_buffer_config["beta"],
            sample_episodes=True,
        )
        sample_seconds = time.perf_counter() - sample_started
        timesteps = {NUM_ENV_STEPS_SAMPLED_LIFETIME: batch_size}

        torch.cuda.reset_peak_memory_stats()
        # The first pass includes any shape-specific compilation and allocator
        # warm-up. The second pass is the steady-state measurement.
        algorithm.learner_group.update(
            episodes=episodes,
            timesteps=timesteps,
            num_epochs=1,
            minibatch_size=minibatch_size,
            shuffle_batch_per_epoch=True,
        )
        measured_started = time.perf_counter()
        algorithm.learner_group.update(
            episodes=episodes,
            timesteps=timesteps,
            num_epochs=1,
            minibatch_size=minibatch_size,
            shuffle_batch_per_epoch=True,
        )
        update_seconds = time.perf_counter() - measured_started
        total_memory = int(
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        )
        peak_reserved = int(torch.cuda.max_memory_reserved())
        result.update(
            {
                "status": "completed",
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_total_memory_bytes": total_memory,
                "sample_seconds": sample_seconds,
                "steady_update_seconds": update_seconds,
                "steady_transitions_per_second": batch_size / update_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": peak_reserved,
                "peak_cuda_reserved_fraction": peak_reserved / total_memory,
            }
        )
    except torch.OutOfMemoryError as error:
        result.update(
            {
                "status": "oom",
                "error": str(error),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    except BaseException as error:
        result.update(
            {
                "status": (
                    "oom" if "out of memory" in str(error).lower() else "failed"
                ),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        if algorithm is not None:
            algorithm.stop()
        result["process_seconds"] = time.perf_counter() - started
        _write_json(output, result)
    return 0 if result["status"] == "completed" else 1


def _run_child(
    *,
    batch_size: int,
    output: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        (
            "experiments.factored_representations_reproduction_"
            "SAC_cycle_2_2026_08.benchmark_minibatch_capacity"
        ),
        "--single",
        "--batch-size",
        str(batch_size),
        "--output",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if output.is_file():
            result = json.loads(output.read_text())
        else:
            result = {
                "batch_size": batch_size,
                "status": "failed",
                "error": "child produced no result file",
            }
        result["child_return_code"] = completed.returncode
        if completed.returncode != 0:
            result["child_log_tail"] = (
                completed.stdout + "\n" + completed.stderr
            ).splitlines()[-40:]
        return result
    except subprocess.TimeoutExpired as error:
        return {
            "batch_size": batch_size,
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "child_log_tail": (
                ((error.stdout or "") + "\n" + (error.stderr or "")).splitlines()[-40:]
            ),
        }


def largest_completed(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the largest successful batch-size result."""

    completed = [result for result in results if result["status"] == "completed"]
    return (
        max(completed, key=lambda result: int(result["batch_size"]))
        if completed
        else None
    )


def run_sweep(
    *,
    candidates: tuple[int, ...],
    output: Path,
    timeout_seconds: int,
) -> int:
    """Run ascending candidates until the first OOM or timeout."""

    summary: dict[str, Any] = {
        "status": "running",
        "minibatch_count": MINIBATCH_COUNT,
        "candidate_batch_sizes": list(candidates),
        "results": [],
    }
    _write_json(output, summary)
    with tempfile.TemporaryDirectory(prefix="sac-minibatch-capacity-") as root:
        root_path = Path(root)
        for batch_size in candidates:
            result = _run_child(
                batch_size=batch_size,
                output=root_path / f"batch_{batch_size}.json",
                timeout_seconds=timeout_seconds,
            )
            summary["results"].append(result)
            _write_json(output, summary)
            if result["status"] in {"oom", "timeout"}:
                break

    winner = largest_completed(summary["results"])
    if winner is not None:
        summary["largest_completed"] = winner
        summary["status"] = "completed"
    else:
        summary["status"] = "failed"
    _write_json(output, summary)
    return 0 if winner is not None else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.single:
        if args.batch_size is None or args.batch_size <= 0:
            raise ValueError("--single requires a positive --batch-size")
        return run_single(batch_size=args.batch_size, output=args.output)
    candidates = tuple(int(value) for value in args.candidates)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("--candidates must contain positive batch sizes")
    if any(value % MINIBATCH_COUNT for value in candidates):
        raise ValueError(f"all candidates must be divisible by {MINIBATCH_COUNT}")
    return run_sweep(
        candidates=candidates,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
