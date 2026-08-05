"""Reproduce the paper's supervised MESS3 belief-fractal experiment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from .analysis import (
    plot_belief_comparison,
    plot_training_curve,
    run_layer_probes,
    write_probe_metrics,
)
from .mess3 import (
    AliasTable,
    enumerate_paths,
    exact_bayesian_loss,
    labeled_operators,
    path_probabilities,
)
from .model import PaperModelConfig, PaperTransformer, parameter_count
from .training import TrainingConfig, load_checkpoint, train


MODEL_CONFIG = PaperModelConfig()
FULL_TRAINING_CONFIG = TrainingConfig()
PROBE_BATCH_SIZE = 4_096
FINAL_MSE_THRESHOLD = 1e-3
VALIDATION_GAP_THRESHOLD_NATS = 0.005
_STREAM_KEYS = {
    "model_initialization": (0,),
    "training_sampling": (1,),
    "probe_split": (2,),
    "plot_sampling": (3,),
}


def _device(context: RunContext) -> torch.device:
    profile = context.hardware or PROFILES["cpu"]
    requested = profile.learner_device
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_model(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _replication_markdown(summary: dict[str, Any]) -> str:
    checks = summary["success_checks"]
    probe = summary["probe"]["layers"][summary["probe"]["headline_layer"]]
    return "\n".join(
        [
            "# MESS3 supervised replication",
            "",
            f"- Analyzed checkpoint: update {summary['analyzed_step']:,}",
            f"- Exact Bayesian floor: {summary['bayesian_floor_nats']:.6f} nats",
            (
                "- Exact validation loss: "
                f"{summary['analyzed_validation_loss_nats']:.6f} nats "
                f"(gap {summary['validation_gap_nats']:+.6f})"
            ),
            (
                "- Final pre-LN affine probe: "
                f"MSE {probe['mse']:.6g}, R² {probe['r2']:.6f}"
            ),
            (
                "- Scientific gate: "
                + (
                    "not applied to smoke mode"
                    if not checks["applicable"]
                    else ("PASS" if checks["passed"] else "FAIL")
                )
            ),
            (
                "- Active optimization: "
                f"{summary['timing']['active_optimization_wall_seconds']:.1f}s "
                f"at {summary['timing']['updates_per_second_active']:.1f} "
                "updates/s"
            ),
            (
                "- End-to-end training wall time: "
                f"{summary['timing']['end_to_end_training_wall_seconds']:.1f}s"
            ),
            (
                "- Probe/plot wall time: "
                f"{summary['timing']['probe_plot_wall_seconds']:.1f}s"
            ),
            (
                "- Total experiment wall time: "
                f"{summary['timing']['experiment_wall_seconds']:.1f}s"
            ),
            "",
            "The model was trained only on next-token cross-entropy. "
            "Belief targets were used only after training by the affine probe.",
            "",
        ]
    )


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the paper replication requires a resolved seed")
    experiment_started = time.monotonic()
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    device = _device(context)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    initialization_seed = seed_sequence_to_int(
        streams["model_initialization"],
        bits=64,
    )
    training_seed = seed_sequence_to_int(
        streams["training_sampling"],
        bits=64,
    )
    probe_seed = seed_sequence_to_int(streams["probe_split"])
    plot_seed = seed_sequence_to_int(streams["plot_sampling"])
    training_config = (
        TrainingConfig.smoke() if context.smoke else FULL_TRAINING_CONFIG
    )

    _seed_model(initialization_seed)
    model = PaperTransformer(MODEL_CONFIG).to(device)
    paths = enumerate_paths(11, device=device)
    probability_dtype = (
        torch.float32 if device.type == "mps" else torch.float64
    )
    operators = labeled_operators(
        device=device,
        dtype=probability_dtype,
    )
    probabilities = path_probabilities(paths, operators=operators)
    probability_mass = float(probabilities.sum().cpu())
    mass_tolerance = 1e-5 if probability_dtype == torch.float32 else 1e-10
    if abs(probability_mass - 1.0) > mass_tolerance:
        raise AssertionError(
            f"length-11 path probabilities sum to {probability_mass}"
        )
    bayesian_floor = exact_bayesian_loss(paths, probabilities)
    alias_table = AliasTable.from_probabilities(
        probabilities,
        device=device,
    )

    outputs.write_json(
        "resolved_recipe.json",
        {
            "paper": "arXiv:2405.15943",
            "objective": "ten shifted next-token cross-entropies",
            "mess3": {
                "x": 0.05,
                "alpha": 0.85,
                "stationary_start": True,
                "labeled_operators": operators.cpu().tolist(),
                "enumerated_length_11_paths": len(paths),
                "path_probability_mass": probability_mass,
            },
            "model": {
                **MODEL_CONFIG.to_dict(),
                "parameter_count": parameter_count(model),
                "residual_probe_location": "block_3 pre-final-LayerNorm",
            },
            "training": training_config.to_dict(),
            "bayesian_floor_nats": bayesian_floor,
            "runtime": {
                "seed": context.seed,
                "device": str(device),
                "smoke": context.smoke,
                "compiled_training": device.type == "cuda",
                "compile_mode": (
                    "reduce-overhead" if device.type == "cuda" else None
                ),
                "compile_fullgraph": device.type == "cuda",
            },
        },
    )

    history, training_summary, analyzed_checkpoint = train(
        model=model,
        paths=paths,
        probabilities=probabilities,
        alias_table=alias_table,
        device=device,
        seed=training_seed,
        config=training_config,
        outputs=outputs,
        resume_from=context.resume_from,
    )
    del alias_table

    _seed_model(initialization_seed)
    analyzed_model = PaperTransformer(MODEL_CONFIG).to(device)
    load_checkpoint(
        analyzed_checkpoint,
        model=analyzed_model,
        optimizer=None,
        generator=None,
        device=device,
    )
    probe_started = time.monotonic()
    contexts = enumerate_paths(10)
    probe, target_beliefs, decoded_beliefs = run_layer_probes(
        analyzed_model,
        contexts,
        seed=probe_seed,
        batch_size=(
            1_024 if context.smoke else PROBE_BATCH_SIZE
        ),
    )
    write_probe_metrics(context.results_dir / "probe_metrics.json", probe)
    headline = probe["layers"][probe["headline_layer"]]
    plot_belief_comparison(
        target_beliefs,
        decoded_beliefs,
        path=context.results_dir / "belief_simplex_comparison.png",
        mse=headline["mse"],
        r2=headline["r2"],
        seed=plot_seed,
    )
    plot_training_curve(
        history,
        floor_nats=bayesian_floor,
        path=context.results_dir / "training_validation_curve.png",
    )
    probe_plot_wall = time.monotonic() - probe_started

    validation_gap = (
        training_summary["analyzed_validation_loss_nats"] - bayesian_floor
    )
    checks_applicable = not context.smoke
    checks_passed = (
        headline["mse"] <= FINAL_MSE_THRESHOLD
        and validation_gap <= VALIDATION_GAP_THRESHOLD_NATS
    )
    summary = {
        **training_summary,
        "bayesian_floor_nats": bayesian_floor,
        "validation_gap_nats": validation_gap,
        "probe": probe,
        "success_checks": {
            "applicable": checks_applicable,
            "final_pre_ln_mse_threshold": FINAL_MSE_THRESHOLD,
            "validation_gap_threshold_nats": (
                VALIDATION_GAP_THRESHOLD_NATS
            ),
            "passed": checks_passed if checks_applicable else None,
        },
        "timing": {
            "active_optimization_wall_seconds": training_summary[
                "active_optimization_wall_seconds"
            ],
            "end_to_end_training_wall_seconds": training_summary[
                "end_to_end_training_wall_seconds"
            ],
            "updates_per_second_active": training_summary[
                "updates_per_second_active"
            ],
            "sequences_per_second_active": training_summary[
                "sequences_per_second_active"
            ],
            "target_tokens_per_second_active": training_summary[
                "target_tokens_per_second_active"
            ],
            "probe_plot_wall_seconds": probe_plot_wall,
            "experiment_wall_seconds": time.monotonic() - experiment_started,
        },
        "outputs": {
            "comparison_figure": str(
                context.results_dir / "belief_simplex_comparison.png"
            ),
            "training_curve": str(
                context.results_dir / "training_validation_curve.png"
            ),
            "probe_metrics": str(
                context.results_dir / "probe_metrics.json"
            ),
            "analyzed_checkpoint": str(analyzed_checkpoint),
        },
    }
    outputs.write_json("summary.json", summary)
    (context.results_dir / "replication_summary.md").write_text(
        _replication_markdown(summary)
    )
    return summary
