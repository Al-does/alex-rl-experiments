"""Isolated CUDA batch-size sweep for the three-factor PPO+CE condition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    build_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES


DEFAULT_CANDIDATES = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
CONDITION = "ppo_aux_ce"
FACTOR_COUNT = 3
TARGET_ENV_STEPS = 10_000_000


def _metric(metrics: dict[str, Any], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return float(direct)
    value: Any = metrics
    for part in path.split("/"):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _benchmark_context(root: Path) -> RunContext:
    return RunContext(
        experiment_dir=Path(__file__).parent,
        results_dir=root / "results",
        artifacts_dir=root / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090"],
    )


def run_single(
    *,
    batch_size: int,
    compile_learner: bool,
    iterations: int,
    output: Path,
) -> int:
    """Run one candidate in its own process so an OOM cannot poison the sweep."""

    result: dict[str, Any] = {
        "batch_size": batch_size,
        "train_batch_size_per_learner": batch_size,
        "minibatch_size": batch_size,
        "condition": CONDITION,
        "factor_count": FACTOR_COUNT,
        "compile_learner": compile_learner,
        "requested_iterations": iterations,
        "status": "failed",
    }
    algorithm = None
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        context = _benchmark_context(output.parent / "runtime")
        config = build_config(
            context,
            factor_count=FACTOR_COUNT,
            condition=CONDITION,
        )
        config.callbacks(on_algorithm_init=None, on_train_result=None)
        config.training(
            train_batch_size_per_learner=batch_size,
            minibatch_size=batch_size,
        )
        config.framework(
            torch_compile_learner=compile_learner,
            torch_compile_worker=False,
        )
        build_started = time.perf_counter()
        algorithm = config.build_algo()
        result["build_seconds"] = time.perf_counter() - build_started

        rows = []
        previous_steps = 0.0
        for iteration in range(1, iterations + 1):
            torch.cuda.reset_peak_memory_stats()
            iteration_started = time.perf_counter()
            metrics = algorithm.train()
            duration = time.perf_counter() - iteration_started
            lifetime_steps = _metric(
                metrics,
                "env_runners/num_env_steps_sampled_lifetime",
            )
            if lifetime_steps is None:
                lifetime_steps = _metric(
                    metrics,
                    "num_env_steps_sampled_lifetime",
                )
            if lifetime_steps is None:
                raise RuntimeError("RLlib did not report sampled environment steps")
            sampled_steps = lifetime_steps - previous_steps
            previous_steps = lifetime_steps
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            rows.append(
                {
                    "iteration": iteration,
                    "duration_seconds": duration,
                    "sampled_env_steps": int(sampled_steps),
                    "end_to_end_steps_per_second": sampled_steps / duration,
                    "sampling_seconds": _metric(
                        metrics,
                        "timers/env_runner_sampling_timer",
                    ),
                    "learner_update_seconds": _metric(
                        metrics,
                        "timers/learner_update_timer",
                    ),
                    "peak_cuda_allocated_bytes": peak_allocated,
                    "peak_cuda_reserved_bytes": peak_reserved,
                }
            )
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_memory = int(properties.total_memory)
        steady_rows = rows[1:] if len(rows) > 1 else rows
        steady_steps = sum(row["sampled_env_steps"] for row in steady_rows)
        steady_seconds = sum(row["duration_seconds"] for row in steady_rows)
        result.update(
            {
                "status": "completed",
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_total_memory_bytes": total_memory,
                "iterations": rows,
                "steady_state_excludes_first_iteration": len(rows) > 1,
                "steady_state_steps_per_second": steady_steps / steady_seconds,
                "max_peak_cuda_allocated_bytes": max(
                    row["peak_cuda_allocated_bytes"] for row in rows
                ),
                "max_peak_cuda_reserved_bytes": max(
                    row["peak_cuda_reserved_bytes"] for row in rows
                ),
                "max_peak_reserved_fraction": max(
                    row["peak_cuda_reserved_bytes"] for row in rows
                )
                / total_memory,
            }
        )
    except torch.OutOfMemoryError as error:
        result.update(
            {
                "status": "oom",
                "error": str(error),
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    except BaseException as error:
        result.update(
            {
                "status": (
                    "oom"
                    if "out of memory" in str(error).lower()
                    else "failed"
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


def choose_finalists(eager_results: list[dict[str, Any]]) -> list[int]:
    """Choose the two fastest safe eager candidates for compiled confirmation."""

    safe = [
        result
        for result in eager_results
        if result.get("status") == "completed"
        and float(result.get("max_peak_reserved_fraction", 1.0)) <= 0.90
    ]
    ranked = sorted(
        safe,
        key=lambda result: (
            float(result["steady_state_steps_per_second"]),
            int(result["batch_size"]),
        ),
        reverse=True,
    )
    return [int(result["batch_size"]) for result in ranked[:2]]


def recommendation(
    compiled_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the highest-throughput compiled candidate with a memory margin."""

    safe = [
        result
        for result in compiled_results
        if result.get("status") == "completed"
        and float(result.get("max_peak_reserved_fraction", 1.0)) <= 0.90
    ]
    if not safe:
        return None
    winner = max(
        safe,
        key=lambda result: float(result["steady_state_steps_per_second"]),
    )
    throughput = float(winner["steady_state_steps_per_second"])
    first = winner["iterations"][0]
    remaining_steps = max(
        TARGET_ENV_STEPS - int(first["sampled_env_steps"]),
        0,
    )
    estimated_seconds = (
        float(winner["build_seconds"])
        + float(first["duration_seconds"])
        + remaining_steps / throughput
    )
    return {
        "batch_size": int(winner["batch_size"]),
        "train_batch_size_per_learner": int(winner["batch_size"]),
        "minibatch_size": int(winner["batch_size"]),
        "steady_state_steps_per_second": throughput,
        "peak_cuda_reserved_bytes": int(
            winner["max_peak_cuda_reserved_bytes"]
        ),
        "peak_cuda_reserved_fraction": float(
            winner["max_peak_reserved_fraction"]
        ),
        "estimated_10m_env_steps_seconds": estimated_seconds,
        "estimated_10m_env_steps_hours": estimated_seconds / 3600.0,
        "estimate_method": (
            "one measured build/compile iteration plus steady-state throughput "
            "for the remaining environment steps"
        ),
    }


def _run_child(
    *,
    batch_size: int,
    compile_learner: bool,
    iterations: int,
    output: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        (
            "experiments.factored_representations_reproduction_PPO_2026_08."
            "benchmark_batch_size"
        ),
        "--single",
        "--batch-size",
        str(batch_size),
        "--iterations",
        str(iterations),
        "--output",
        str(output),
    ]
    if compile_learner:
        command.append("--compile-learner")
    environment = dict(os.environ)
    environment.setdefault("RAY_DEDUP_LOGS", "1")
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=environment,
            check=False,
        )
        if output.is_file():
            result = json.loads(output.read_text())
        else:
            result = {
                "batch_size": batch_size,
                "compile_learner": compile_learner,
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
            "compile_learner": compile_learner,
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "child_log_tail": (
                ((error.stdout or "") + "\n" + (error.stderr or "")).splitlines()[
                    -40:
                ]
            ),
        }


def run_sweep(
    *,
    candidates: tuple[int, ...],
    output: Path,
    timeout_seconds: int,
) -> int:
    """Run eager capacity sweep, then compiled steady-state finalists."""

    summary: dict[str, Any] = {
        "status": "running",
        "condition": CONDITION,
        "factor_count": FACTOR_COUNT,
        "candidate_batch_sizes": list(candidates),
        "memory_safety_limit": 0.90,
        "eager_capacity_sweep": [],
        "compiled_finalists": [],
        "target_env_steps_for_estimate": TARGET_ENV_STEPS,
    }
    _write_json(output, summary)
    with tempfile.TemporaryDirectory(prefix="factored-batch-benchmark-") as root:
        temporary_root = Path(root)
        for batch_size in candidates:
            result = _run_child(
                batch_size=batch_size,
                compile_learner=False,
                iterations=1,
                output=temporary_root / f"eager_{batch_size}.json",
                timeout_seconds=timeout_seconds,
            )
            summary["eager_capacity_sweep"].append(result)
            _write_json(output, summary)
            if result["status"] in {"oom", "timeout"}:
                break

        finalists = choose_finalists(summary["eager_capacity_sweep"])
        summary["selected_compiled_finalists"] = finalists
        for batch_size in finalists:
            result = _run_child(
                batch_size=batch_size,
                compile_learner=True,
                iterations=3,
                output=temporary_root / f"compiled_{batch_size}.json",
                timeout_seconds=timeout_seconds,
            )
            summary["compiled_finalists"].append(result)
            _write_json(output, summary)

    selected = recommendation(summary["compiled_finalists"])
    summary["recommendation"] = selected
    summary["status"] = "completed" if selected is not None else "failed"
    _write_json(output, summary)
    return 0 if selected is not None else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--compile-learner", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
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
        if args.iterations <= 0:
            raise ValueError("--iterations must be positive")
        return run_single(
            batch_size=args.batch_size,
            compile_learner=args.compile_learner,
            iterations=args.iterations,
            output=args.output,
        )
    candidates = tuple(int(value) for value in args.candidates)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("--candidates must contain positive batch sizes")
    return run_sweep(
        candidates=candidates,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
