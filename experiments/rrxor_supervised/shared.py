"""Paper-faithful supervised RRXOR training and analysis workflow."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from experiments.mess3_supervised.paper_supervised_replication.training import (
    TrainingConfig,
    load_checkpoint,
    train,
)
from experiments.rrxor_token_guess_cycle_1.process import (
    RRXOR_STATIONARY,
    RRXOR_TENSOR,
    reachable_beliefs,
)

from .analysis import (
    exact_task_metrics,
    plot_belief_geometry,
    plot_layer_mse,
    plot_pairwise_distances,
    plot_training_curve,
    run_belief_probes,
    write_json,
)
from .model import MODEL_CONFIG, RRXORPaperTransformer, parameter_count
from .process import (
    AliasTable,
    exact_bayesian_loss,
    possible_paths,
)


FULL_TRAINING_CONFIG = TrainingConfig(
    total_steps=1_000_000,
    analyzed_step=1_000_000,
    batch_size=64,
    optimizer_name="sgd",
    learning_rate=0.01,
    weight_decay=0.0,
    momentum=0.0,
    log_every=1_000,
    checkpoint_every=10_000,
    retain_periodic_checkpoints=False,
    validation_every=50_000,
    validation_batch_size=4_096,
)
SMOKE_TRAINING_CONFIG = TrainingConfig(
    total_steps=100,
    analyzed_step=100,
    batch_size=64,
    optimizer_name="sgd",
    learning_rate=0.01,
    weight_decay=0.0,
    momentum=0.0,
    log_every=10,
    checkpoint_every=50,
    retain_periodic_checkpoints=False,
    validation_every=50,
    validation_batch_size=2_048,
)
PROBE_BATCH_SIZE = 4_096
_STREAM_KEYS = {
    "model_initialization": (0,),
    "training_sampling": (1,),
    "probe_split": (2,),
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


def _headline(report: dict[str, Any]) -> dict[str, Any]:
    representations = report["geometry"]["representations"]
    layer_metrics = {
        name: {
            "mse": record["metrics"]["mse"],
            "r_squared": record["metrics"]["r_squared"],
        }
        for name, record in representations.items()
        if name.startswith("layer_")
    }
    concatenated = representations["concatenated_layers"]["metrics"]
    final_norm = representations["post_final_layer_norm"]["metrics"]
    return {
        "layers": layer_metrics,
        "concatenated_layers": {
            "mse": concatenated["mse"],
            "r_squared": concatenated["r_squared"],
            "prediction_null_contrasts": concatenated["contrasts"],
        },
        "post_final_layer_norm": {
            "mse": final_norm["mse"],
            "r_squared": final_norm["r_squared"],
        },
        "paper_pairwise_distance_control": report[
            "paper_pairwise_distance_control"
        ],
    }


def _replication_markdown(summary: dict[str, Any]) -> str:
    geometry = summary["probe_headline"]
    concat = geometry["concatenated_layers"]
    pairwise = geometry["paper_pairwise_distance_control"]
    checks = summary["success_checks"]
    return "\n".join(
        [
            "# RRXOR supervised paper replication",
            "",
            f"- Analyzed checkpoint: update {summary['analyzed_step']:,}",
            (
                "- Exact length-10-sequence Bayesian floor: "
                f"{summary['bayesian_floor_nats']:.6f} nats"
            ),
            (
                "- Exact validation loss: "
                f"{summary['task']['cross_entropy_nats']:.6f} nats "
                f"(gap {summary['validation_gap_nats']:+.6f})"
            ),
            (
                "- Concatenated-layer affine probe: "
                f"MSE {concat['mse']:.6g}, "
                f"R² {concat['r_squared']:.6f}"
            ),
            (
                "- Pairwise-distance R²: belief "
                f"{pairwise['belief_distance_to_decoded_distance_r_squared']:.4f}, "
                "next-token "
                f"{pairwise['next_token_distance_to_decoded_distance_r_squared']:.4f}"
            ),
            (
                "- Scientific checks: "
                + (
                    "not applied to smoke mode"
                    if not checks["applicable"]
                    else ("PASS" if checks["passed"] else "FAIL")
                )
            ),
            "",
            "Training used only shifted next-token cross-entropy. "
            "Exact beliefs were used only for post-training analysis.",
            "",
        ]
    )


def run_condition(context: RunContext):
    """Train and analyze the supervised RRXOR paper reproduction."""

    if context.seed is None:
        raise ValueError("the paper reproduction requires a resolved seed")
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
    training_config = (
        SMOKE_TRAINING_CONFIG if context.smoke else FULL_TRAINING_CONFIG
    )

    _seed_model(initialization_seed)
    model = RRXORPaperTransformer(MODEL_CONFIG).to(device)
    probability_dtype = (
        torch.float32 if device.type == "mps" else torch.float64
    )
    paths, probabilities = possible_paths(
        10,
        device=device,
        dtype=probability_dtype,
    )
    probability_mass = float(probabilities.sum().cpu())
    mass_tolerance = 1e-5 if probability_dtype == torch.float32 else 1e-10
    if abs(probability_mass - 1.0) > mass_tolerance:
        raise AssertionError(
            f"length-10 path probabilities sum to {probability_mass}"
        )
    bayesian_floor = exact_bayesian_loss(paths, probabilities)
    alias_table = AliasTable.from_probabilities(
        probabilities,
        device=device,
    )
    reachable = reachable_beliefs()
    if reachable.shape != (36, 5):
        raise AssertionError(
            f"expected 36 RRXOR beliefs, found {reachable.shape}"
        )

    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "rrxor_supervised",
            "condition": "paper_supervised_replication",
            "paper": "arXiv:2405.15943",
            "hypothesis": (
                "next-token cross-entropy learns RRXOR belief geometry that "
                "is weak in individual residual layers and recoverable from "
                "their concatenation"
            ),
            "objective": {
                "type": "next_token_cross_entropy",
                "sequence_length": 10,
                "shifted_target_positions": 9,
                "belief_supervision": False,
            },
            "rrxor": {
                "edge_transition_matrices": RRXOR_TENSOR.tolist(),
                "stationary_initial_distribution": (
                    RRXOR_STATIONARY.tolist()
                ),
                "reachable_belief_states": int(len(reachable)),
                "positive_probability_length_10_paths": int(len(paths)),
                "path_probability_mass": probability_mass,
            },
            "model": {
                **MODEL_CONFIG.to_dict(),
                "parameter_count": parameter_count(model),
                "residual_probe_locations": [
                    "post-block residual for each of four blocks",
                    "concatenation of all four post-block residuals",
                    "post-final-LayerNorm residual",
                ],
            },
            "training": training_config.to_dict(),
            "analysis": {
                "contexts": (
                    "all positive-probability binary length-10 strings"
                ),
                "positions": "all ten positions in every context",
                "probe": "grouped held-out affine regression",
                "fit_fraction": 0.2,
                "controls": [
                    "initialization",
                    "current token",
                    "three-token suffix",
                    "exact next-token probabilities and logs",
                    "prediction-null belief contrasts",
                    "permuted targets",
                    "Gaussian features",
                    "suffix-matched features",
                    "paper pairwise-distance comparison",
                ],
            },
            "bayesian_floor_nats": bayesian_floor,
            "runtime": {
                "seed": context.seed,
                "device": str(device),
                "smoke": context.smoke,
                "compiled_training": device.type == "cuda",
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

    initialization_model = RRXORPaperTransformer(MODEL_CONFIG).to(device)
    load_checkpoint(
        outputs.checkpoints_dir / "step_0000000.pt",
        model=initialization_model,
        optimizers=None,
        generator=None,
        device=device,
    )
    analyzed_model = RRXORPaperTransformer(MODEL_CONFIG).to(device)
    load_checkpoint(
        analyzed_checkpoint,
        model=analyzed_model,
        optimizers=None,
        generator=None,
        device=device,
    )
    task_metrics = exact_task_metrics(
        analyzed_model,
        paths,
        probabilities,
        batch_size=training_config.validation_batch_size,
    )

    probe_started = time.monotonic()
    contexts, _ = possible_paths(10)
    probe_report, battery, test_beliefs, pairwise_arrays = run_belief_probes(
        analyzed_model,
        initialization_model,
        contexts,
        seed=probe_seed,
        batch_size=1_024 if context.smoke else PROBE_BATCH_SIZE,
        smoke=context.smoke,
    )
    write_json(context.results_dir / "probe_metrics.json", probe_report)
    raw_arrays = {
        "test_beliefs": test_beliefs,
        **{
            f"prediction_{name}": prediction
            for name, prediction in battery.predictions.items()
        },
        **{
            f"baseline_{name.replace('/', '__')}": prediction
            for name, prediction in battery.baseline_predictions.items()
        },
    }
    np.savez_compressed(
        context.artifacts_dir / "probe_arrays.npz",
        **raw_arrays,
    )
    plot_belief_geometry(
        test_beliefs,
        battery.predictions["concatenated_layers"],
        path=context.results_dir / "belief_geometry.png",
    )
    plot_layer_mse(
        probe_report,
        path=context.results_dir / "layer_mse.png",
    )
    plot_pairwise_distances(
        pairwise_arrays,
        probe_report["paper_pairwise_distance_control"],
        path=context.results_dir / "pairwise_distances.png",
    )
    plot_training_curve(
        history,
        floor_nats=bayesian_floor,
        path=context.results_dir / "training_validation_curve.png",
    )
    probe_wall_seconds = time.monotonic() - probe_started

    validation_gap = task_metrics["cross_entropy_nats"] - bayesian_floor
    headline = _headline(probe_report)
    layer_mses = [
        metrics["mse"] for metrics in headline["layers"].values()
    ]
    pairwise = headline["paper_pairwise_distance_control"]
    checks_applicable = not context.smoke
    checks_passed = (
        validation_gap <= 0.005
        and headline["concatenated_layers"]["mse"] < min(layer_mses)
        and pairwise["belief_distance_to_decoded_distance_r_squared"]
        > pairwise["next_token_distance_to_decoded_distance_r_squared"]
    )
    summary = {
        **training_summary,
        "study": "rrxor_supervised",
        "condition": "paper_supervised_replication",
        "bayesian_floor_nats": bayesian_floor,
        "validation_gap_nats": validation_gap,
        "task": task_metrics,
        "probe_headline": headline,
        "success_checks": {
            "applicable": checks_applicable,
            "validation_gap_threshold_nats": 0.005,
            "concatenated_mse_below_each_individual_layer": (
                headline["concatenated_layers"]["mse"] < min(layer_mses)
            ),
            "belief_distances_outperform_next_token_distances": (
                pairwise[
                    "belief_distance_to_decoded_distance_r_squared"
                ]
                > pairwise[
                    "next_token_distance_to_decoded_distance_r_squared"
                ]
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
            "probe_plot_wall_seconds": probe_wall_seconds,
            "experiment_wall_seconds": time.monotonic() - experiment_started,
        },
        "outputs": {
            "probe_metrics": str(
                context.results_dir / "probe_metrics.json"
            ),
            "raw_probe_arrays": str(
                context.artifacts_dir / "probe_arrays.npz"
            ),
            "belief_geometry": str(
                context.results_dir / "belief_geometry.png"
            ),
            "layer_mse": str(context.results_dir / "layer_mse.png"),
            "pairwise_distances": str(
                context.results_dir / "pairwise_distances.png"
            ),
            "training_curve": str(
                context.results_dir / "training_validation_curve.png"
            ),
            "analyzed_checkpoint": str(analyzed_checkpoint),
        },
    }
    outputs.write_json("summary.json", summary)
    (context.results_dir / "replication_summary.md").write_text(
        _replication_markdown(summary)
    )
    return summary
