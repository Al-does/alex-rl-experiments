"""Shared paper recipe for the two supervised checkpoint experiments."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_PARAMETERS,
    nonergodic_mess3_model,
)

from .analysis import AnalysisConfig, run_checkpoint_analysis
from .data import BOS_TOKEN, EPISODE_LENGTH, VOCAB_SIZE
from .model import ModelConfig, NonergodicMess3Transformer, parameter_count
from .training import TrainingConfig, train


MODEL_CONFIG = ModelConfig()
_STREAM_KEYS = {
    "model_initialization": (0,),
    "training_sampling": (1,),
    "validation_sampling": (2,),
    "probe_fit_sampling": (3,),
    "probe_test_sampling": (4,),
    "probe_analysis": (5,),
}


def _device(context: RunContext) -> torch.device:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but CUDA is unavailable")
        return torch.device("cuda")
    if (
        profile.learner_device == "mps"
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _markdown(summary: dict[str, Any]) -> str:
    headline = summary["analysis"]["headline"]
    checks = summary["success_checks"]
    return "\n".join(
        [
            f"# Nonergodic MESS3 supervised: {summary['condition']}",
            "",
            f"- Analyzed step: {summary['training']['analyzed_step']:,}",
            (
                "- Held-out loss gap: "
                f"{summary['training']['excess_loss_nats']:.6g} nats/token"
            ),
            (
                "- Block 3 weighted-belief probe: "
                f"R² {headline['block_3']['r_squared']:.6f}"
            ),
            (
                "- Block 4 weighted-belief probe: "
                f"R² {headline['block_4']['r_squared']:.6f}"
            ),
            (
                "- Paper-target checks: "
                + (
                    "not applied to smoke mode"
                    if not checks["applicable"]
                    else ("PASS" if checks["passed"] else "FAIL")
                )
            ),
            "",
            "The model is trained only on next-token cross-entropy. "
            "Exact six-state beliefs are used only for post-hoc probes.",
            "",
        ]
    )


def run_condition(
    context: RunContext,
    *,
    condition: str,
    full_training_config: TrainingConfig,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("this supervised experiment requires a resolved seed")
    started_at = time.monotonic()
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    device = _device(context)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    seeds = {
        name: seed_sequence_to_int(
            stream,
            bits=64 if name != "probe_analysis" else 32,
        )
        for name, stream in streams.items()
    }
    training_config = (
        TrainingConfig.smoke() if context.smoke else full_training_config
    )
    analysis_config = (
        AnalysisConfig.smoke() if context.smoke else AnalysisConfig()
    )

    _seed_torch(seeds["model_initialization"])
    model = NonergodicMess3Transformer(MODEL_CONFIG).to(device)
    hmm = nonergodic_mess3_model()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "source": {
                "article": "https://simplex.pub/nonergodic-geometry/",
                "section": "Training details",
                "condition": condition,
            },
            "hypothesis": (
                "next-token cross-entropy on a fixed-component nonergodic "
                "composition produces a linearly readable weighted six-state "
                "belief geometry in the residual stream"
            ),
            "data": {
                "component_prior": [0.5, 0.5],
                "component_parameters": COMPONENT_PARAMETERS,
                "component_fixed_for_sequence": True,
                "bos_token": BOS_TOKEN,
                "emissions_per_sequence": EPISODE_LENGTH,
                "sequence_length": EPISODE_LENGTH + 1,
                "vocabulary_size": VOCAB_SIZE,
                "edge_transition_matrices": (
                    hmm.edge_transition_matrices.tolist()
                ),
            },
            "model": {
                **MODEL_CONFIG.to_dict(),
                "implementation": (
                    "experiment-local PyTorch equivalent of the article's "
                    "TransformerLens HookedTransformer"
                ),
                "parameter_count": parameter_count(model),
                "probe_locations": model.residual_names,
                "article_headline_location": "block_3",
            },
            "objective": {
                "type": "shifted next-token cross-entropy",
                "target_positions_per_sequence": EPISODE_LENGTH,
                "belief_supervision": False,
            },
            "training": training_config.to_dict(),
            "analysis": analysis_config.to_dict(),
            "runtime": {
                "seed": context.seed,
                "derived_seeds": seeds,
                "device": str(device),
                "smoke": context.smoke,
            },
        },
    )

    history, training_summary, checkpoint_paths = train(
        model=model,
        device=device,
        training_seed=seeds["training_sampling"],
        validation_seed=seeds["validation_sampling"],
        config=training_config,
        outputs=outputs,
        resume_from=context.resume_from,
    )
    analysis_started = time.monotonic()
    analysis = run_checkpoint_analysis(
        checkpoint_paths=checkpoint_paths,
        history=history,
        model_config=MODEL_CONFIG,
        device=device,
        fit_seed=seeds["probe_fit_sampling"],
        test_seed=seeds["probe_test_sampling"],
        analysis_seed=seeds["probe_analysis"],
        config=analysis_config,
        results_dir=context.results_dir,
    )
    analysis_seconds = time.monotonic() - analysis_started
    headline = analysis["headline"]
    checks_applicable = not context.smoke
    checks = {
        "loss_gap_at_most_5e-4": (
            training_summary["excess_loss_nats"] <= 5e-4
        ),
        "block_3_r_squared_at_least_0_97": (
            headline["block_3"]["r_squared"] is not None
            and headline["block_3"]["r_squared"] >= 0.97
        ),
        "block_4_r_squared_at_least_0_98": (
            headline["block_4"]["r_squared"] is not None
            and headline["block_4"]["r_squared"] >= 0.98
        ),
    }
    summary = {
        "condition": condition,
        "training": training_summary,
        "analysis": analysis,
        "success_checks": {
            "applicable": checks_applicable,
            "article_reports": {
                "step_10000_loss_gap": "a few 1e-4 nats/token",
                "step_45000_loss_gap": "about 1e-4 nats/token",
                "block_3_r_squared": "about 0.985",
                "block_4_r_squared": "about 0.99",
            },
            "checks": checks,
            "passed": all(checks.values()) if checks_applicable else None,
        },
        "timing": {
            **{
                key: training_summary[key]
                for key in (
                    "active_optimization_wall_seconds",
                    "end_to_end_training_wall_seconds",
                    "updates_per_second_active",
                    "sequences_per_second_active",
                    "target_tokens_per_second_active",
                )
            },
            "analysis_wall_seconds": analysis_seconds,
            "experiment_wall_seconds": time.monotonic() - started_at,
        },
        "outputs": {
            "analysis": str(context.results_dir / "analysis.json"),
            "geometry": str(context.results_dir / "block_3_geometry.png"),
            "activation_variance": str(
                context.results_dir / "block_3_activation_variance.png"
            ),
            "training_probe_curves": str(
                context.results_dir / "training_and_probe_curves.png"
            ),
            "checkpoint_directory": str(outputs.checkpoints_dir),
        },
    }
    outputs.write_json("analysis.json", analysis)
    outputs.write_json("summary.json", summary)
    (context.results_dir / "findings.md").write_text(_markdown(summary))
    return summary
