"""Pure next-token recipe over two and three independent MESS3 factors."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any

import torch

from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from .analysis import analyze_checkpoint, plot_probe_trajectory
from .model import FactoredNextTokenTransformer, NextTokenModelConfig
from .process import (
    FACTOR_COUNTS,
    MESS3_ALPHA,
    MESS3_X,
    SEQUENCE_LENGTH,
    joint_token_count,
)
from .training import TrainingConfig, train


FULL_TRAINING_CONFIG = TrainingConfig()
SMOKE_TRAINING_CONFIG = TrainingConfig.smoke()


def _training_config_for_run(*, smoke: bool) -> TrainingConfig:
    """Resolve the training budget, honoring optional launch env overrides."""

    if smoke:
        return SMOKE_TRAINING_CONFIG
    target_sequences = os.environ.get("FACTORED_NEXT_TOKEN_TARGET_SEQUENCES")
    if target_sequences is not None:
        sequences = int(target_sequences)
        if sequences <= 0:
            raise ValueError("FACTORED_NEXT_TOKEN_TARGET_SEQUENCES must be positive")
        if sequences % FULL_TRAINING_CONFIG.batch_size != 0:
            raise ValueError(
                "FACTORED_NEXT_TOKEN_TARGET_SEQUENCES must divide batch size "
                f"{FULL_TRAINING_CONFIG.batch_size}"
            )
        return replace(
            FULL_TRAINING_CONFIG,
            total_updates=sequences // FULL_TRAINING_CONFIG.batch_size,
        )
    total_updates = os.environ.get("FACTORED_NEXT_TOKEN_TOTAL_UPDATES")
    if total_updates is not None:
        return replace(
            FULL_TRAINING_CONFIG,
            total_updates=int(total_updates),
        )
    return FULL_TRAINING_CONFIG
_STREAM_KEYS = {
    "model_initialization": (700,),
    "training_sampling": (701,),
    "validation_sampling": (702,),
    "checkpoint_probes": (703,),
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


def _resolved_recipe(
    *,
    factor_count: int,
    context: RunContext,
    device: torch.device,
    model_config: NextTokenModelConfig,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    return {
        "paper": "Transformers learn factored representations (arXiv:2602.02385)",
        "source_implementation": "https://github.com/Astera-org/factored-reps",
        "condition": "pure_next_token_prediction",
        "factor_count": factor_count,
        "factors": {
            "type": "independent_MESS3",
            "alpha": MESS3_ALPHA,
            "x": MESS3_X,
            "state_dimension": 3,
            "subtoken_cardinality": 3,
        },
        "joint_token_count": joint_token_count(factor_count),
        "sequence": {
            "generated_tokens": SEQUENCE_LENGTH,
            "input_shift": "[BOS, x1, ..., x7] -> [x1, ..., x8]",
            "context_capacity": model_config.context_length,
        },
        "objective": {
            "type": "cross_entropy",
            "target": "next_joint_token",
            "reinforcement_learning": False,
            "reward": None,
            "value_function": None,
            "policy_loss": None,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": training_config.learning_rate,
            "betas": [training_config.beta_1, training_config.beta_2],
            "epsilon": training_config.epsilon,
            "weight_decay": training_config.weight_decay,
        },
        "model": model_config.to_dict(),
        "head_choice_rationale": (
            "PR #59 reduced the paper's d_model=120 to 64. Four 16-dimensional "
            "heads preserve that controlled adaptation because three heads do "
            "not divide 64."
        ),
        "training": training_config.to_dict(),
        "checkpoint_schedule": "initialization, powers of two, final",
        "analysis": [
            "held-out affine factor-belief regression",
            "activation/factored-target/joint-target CEV",
            "vary-one per-position-centered factor PCA",
            "effective-dimension additivity and principal-angle overlap",
            "rank-two projected factor-belief recovery",
            "joint-token embedding geometry",
            "validation cross-entropy gap to sampled exact Bayes loss",
        ],
        "runtime": {
            "seed": context.seed,
            "smoke": context.smoke,
            "device": str(device),
            "compiled_training": device.type == "cuda",
        },
        "scientific_scope": (
            "This isolates the paper's supervised pretraining mechanism. "
            "There is no Gym environment, action, reward, policy, critic, "
            "RLlib Algorithm, or PPO loss."
        ),
    }


def _checkpoint_update(path: Path) -> int:
    return int(path.stem.removeprefix("update_"))


def run_factor_count(
    context: RunContext,
    *,
    factor_count: int,
) -> dict[str, Any]:
    """Train and analyze one independent-factor next-token condition."""

    if context.seed is None:
        raise ValueError("the next-token study requires a resolved seed")
    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
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
    validation_seed = seed_sequence_to_int(
        streams["validation_sampling"],
        bits=64,
    )
    probe_seed = seed_sequence_to_int(streams["checkpoint_probes"])
    model_config = NextTokenModelConfig(factor_count=factor_count)
    training_config = _training_config_for_run(smoke=context.smoke)
    outputs.write_json(
        "resolved_recipe.json",
        _resolved_recipe(
            factor_count=factor_count,
            context=context,
            device=device,
            model_config=model_config,
            training_config=training_config,
        ),
    )

    _seed_model(initialization_seed)
    model = FactoredNextTokenTransformer(model_config).to(device)
    _, training_summary, _ = train(
        model=model,
        factor_count=factor_count,
        device=device,
        seed=training_seed,
        validation_seed=validation_seed,
        config=training_config,
        outputs=outputs,
        resume_from=context.resume_from,
    )
    checkpoint_paths = sorted(
        outputs.checkpoints_dir.glob("update_*.pt"),
        key=_checkpoint_update,
    )
    reports = []
    for checkpoint in checkpoint_paths:
        update = _checkpoint_update(checkpoint)
        reports.append(
            analyze_checkpoint(
                checkpoint=checkpoint,
                factor_count=factor_count,
                update=update,
                seed=probe_seed,
                smoke=context.smoke,
                device=device,
                results_dir=(
                    context.results_dir
                    / "checkpoint_probes"
                    / f"updates_{update:06d}"
                ),
            )
        )
    trajectory = context.results_dir / "probe_trajectory.png"
    plot_probe_trajectory(
        reports,
        factor_count=factor_count,
        path=trajectory,
    )
    summary = {
        "condition": "pure_next_token_prediction",
        "factor_count": factor_count,
        "seed": context.seed,
        "smoke": context.smoke,
        "training": training_summary,
        "checkpoint_reports": [
            {
                "update": report["update"],
                "path": str(
                    context.results_dir
                    / "checkpoint_probes"
                    / f"updates_{report['update']:06d}"
                    / "probe_battery.json"
                ),
            }
            for report in reports
        ],
        "trajectory_figure": str(trajectory),
    }
    outputs.write_json("summary.json", summary)
    return summary


def run_study(context: RunContext) -> dict[str, Any]:
    """Run the two- and three-factor pure next-token comparison."""

    summaries = {}
    for factor_count in FACTOR_COUNTS:
        factor_context = replace(
            context,
            results_dir=context.results_dir / f"{factor_count}_factors",
            artifacts_dir=context.artifacts_dir / f"{factor_count}_factors",
            resume_from=None,
        )
        summaries[str(factor_count)] = run_factor_count(
            factor_context,
            factor_count=factor_count,
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    result = {
        "condition": "pure_next_token_prediction",
        "seed": context.seed,
        "smoke": context.smoke,
        "factor_conditions": summaries,
    }
    outputs.write_json("study_summary.json", result)
    return result
