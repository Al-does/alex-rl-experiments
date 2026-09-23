"""Shared setup, exact evaluation, and checkpoint probes for both arms."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from .analysis import (
    plot_belief_comparison,
    plot_probe_trajectory,
    plot_training_curve,
    run_layer_probes,
    write_probe_metrics,
)
from .mess3 import (
    PAPER_ALPHA,
    PAPER_X,
    AliasTable,
    bayesian_beliefs,
    enumerate_paths,
    exact_bayesian_loss,
    labeled_operators,
    next_token_probabilities,
    path_probabilities,
)
from .model import (
    PaperActorCritic,
    PaperModelConfig,
    PaperTransformer,
    parameter_count,
)
from .ppo import PPOConfig, train_ppo
from .supervised import TrainingConfig, train


MODEL_CONFIG = PaperModelConfig()
PROBE_BATCH_SIZE = 4_096
# 400 whole length-10 contexts = 4,000 positions for each of fit and test.
PROBE_FIT_SEQUENCES = 400
PROBE_TEST_SEQUENCES = 400
EVALUATION_BATCH_SIZE = 16_384
FINAL_MSE_THRESHOLD = 1e-3
VALIDATION_GAP_THRESHOLD_NATS = 0.005
HEADLINE_LAYER = f"block_{MODEL_CONFIG.n_layers - 1}"
_STREAM_KEYS = {
    "model_initialization": (0,),
    "training_sampling": (1,),
    "probe_split": (2,),
    "plot_sampling": (3,),
}


@dataclass(frozen=True, slots=True)
class RunSetup:
    device: torch.device
    initialization_seed: int
    training_seed: int
    probe_seed: int
    plot_seed: int
    paths: torch.Tensor
    probabilities: torch.Tensor
    alias_table: AliasTable
    bayesian_floor: float
    bayesian_accuracy: float


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


def exact_bayesian_accuracy(
    paths: torch.Tensor,
    probabilities: torch.Tensor,
) -> float:
    """Expected greedy next-token accuracy of the exact Bayesian predictor."""
    operators = labeled_operators(
        device=paths.device,
        dtype=probabilities.dtype,
    )
    beliefs = bayesian_beliefs(paths[:, :-1], operators=operators)
    guesses = next_token_probabilities(beliefs, operators=operators).argmax(-1)
    per_path = (guesses == paths[:, 1:]).to(probabilities.dtype).mean(dim=-1)
    return float((probabilities * per_path).sum() / probabilities.sum())


@torch.no_grad()
def exact_policy_metrics(
    model: PaperTransformer,
    paths: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> dict[str, float]:
    """Exact process-weighted CE, greedy accuracy, and sampled-policy reward."""
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, dtype=probabilities.dtype, device=paths.device)
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        weights = probabilities[start : start + batch_size]
        log_probabilities = F.log_softmax(
            model.forward(batch[:, :-1]).to(probabilities.dtype),
            dim=-1,
        )
        targets = batch[:, 1:].unsqueeze(-1)
        target_log_probs = log_probabilities.gather(-1, targets).squeeze(-1)
        greedy = (log_probabilities.argmax(-1) == batch[:, 1:]).to(
            probabilities.dtype
        )
        per_path = torch.stack(
            [
                -target_log_probs.mean(-1),
                greedy.mean(-1),
                target_log_probs.exp().mean(-1),
            ],
            dim=-1,
        )
        totals += (weights.unsqueeze(-1) * per_path).sum(dim=0)
    if was_training:
        model.train()
    values = (totals / probabilities.sum()).cpu().tolist()
    return {
        "cross_entropy_nats": values[0],
        "greedy_accuracy": values[1],
        "expected_sampled_reward": values[2],
    }


def prepare(context: RunContext) -> RunSetup:
    if context.seed is None:
        raise ValueError("the paper replication requires a resolved seed")
    device = _device(context)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    paths = enumerate_paths(MODEL_CONFIG.context_length + 1, device=device)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    probabilities = path_probabilities(
        paths,
        operators=labeled_operators(device=device, dtype=dtype),
    )
    mass = float(probabilities.sum().cpu())
    if abs(mass - 1.0) > (1e-5 if dtype == torch.float32 else 1e-10):
        raise AssertionError(f"length-11 path probabilities sum to {mass}")
    return RunSetup(
        device=device,
        initialization_seed=seed_sequence_to_int(
            streams["model_initialization"], bits=64
        ),
        training_seed=seed_sequence_to_int(
            streams["training_sampling"], bits=64
        ),
        probe_seed=seed_sequence_to_int(streams["probe_split"]),
        plot_seed=seed_sequence_to_int(streams["plot_sampling"]),
        paths=paths,
        probabilities=probabilities,
        alias_table=AliasTable.from_probabilities(probabilities, device=device),
        bayesian_floor=exact_bayesian_loss(paths, probabilities),
        bayesian_accuracy=exact_bayesian_accuracy(paths, probabilities),
    )


def _recipe(
    context: RunContext,
    setup: RunSetup,
    *,
    arm: str,
    objective: str,
    model: PaperTransformer,
    training: dict[str, Any],
) -> dict[str, Any]:
    return {
        "paper": "arXiv:2405.15943",
        "arm": arm,
        "objective": objective,
        "mess3": {"x": PAPER_X, "alpha": PAPER_ALPHA, "stationary_start": True},
        "data": "fresh stationary length-11 sequences every update/rollout",
        "model": {
            **MODEL_CONFIG.to_dict(),
            "class": type(model).__name__,
            "parameter_count": parameter_count(model),
        },
        "belief_probe": {
            "type": "affine_ols",
            "target": "exact_bayesian_belief_given_context",
            "contexts": "uniform sample of 3^10 length-10 paths, every position",
            "fit_positions": PROBE_FIT_SEQUENCES * MODEL_CONFIG.context_length,
            "test_positions": PROBE_TEST_SEQUENCES * MODEL_CONFIG.context_length,
            "split": "disjoint whole contexts, same split at every checkpoint",
            "headline_layer": f"{HEADLINE_LAYER} (pre-final-LayerNorm)",
        },
        "training": training,
        "bayesian_floor_nats": setup.bayesian_floor,
        "bayesian_greedy_accuracy": setup.bayesian_accuracy,
        "runtime": {
            "seed": context.seed,
            "device": str(setup.device),
            "smoke": context.smoke,
        },
    }


def probe_checkpoints(
    checkpoints: list[tuple[int, Path]],
    *,
    build_model: Callable[[], PaperTransformer],
    setup: RunSetup,
    context: RunContext,
    extra: Callable[[Path], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], Any, Any]]:
    """Probe every retained checkpoint; return the last checkpoint's figure data."""
    if not checkpoints:
        raise RuntimeError("no checkpoints were retained for probing")
    contexts = enumerate_paths(MODEL_CONFIG.context_length)
    batch_size = 1_024 if context.smoke else PROBE_BATCH_SIZE
    trajectory: list[dict[str, Any]] = []
    model = build_model()
    for agent_steps, path in sorted(checkpoints):
        state = torch.load(path, map_location=setup.device, weights_only=False)
        model.load_state_dict(state["model_state"])
        probe, target, decoded = run_layer_probes(
            model,
            contexts,
            seed=setup.probe_seed,
            batch_size=batch_size,
            fit_sequences=PROBE_FIT_SEQUENCES,
            test_sequences=PROBE_TEST_SEQUENCES,
        )
        trajectory.append(
            {
                "agent_steps": agent_steps,
                "checkpoint": path.name,
                "headline_mse": probe["layers"][HEADLINE_LAYER]["mse"],
                "headline_r2": probe["layers"][HEADLINE_LAYER]["r2"],
                "final_ln_r2": probe["layers"]["final_ln"]["r2"],
                "layers": {
                    name: {"mse": metrics["mse"], "r2": metrics["r2"]}
                    for name, metrics in probe["layers"].items()
                },
                **exact_policy_metrics(
                    model, setup.paths, setup.probabilities
                ),
                **({} if extra is None else extra(path)),
            }
        )
    return trajectory, (probe, target, decoded)


def _finish(
    context: RunContext,
    outputs: RunArtifacts,
    setup: RunSetup,
    *,
    arm: str,
    trajectory: list[dict[str, Any]],
    probe_data: tuple[dict[str, Any], Any, Any],
    extra_checks: dict[str, bool],
    extra_summary: dict[str, Any] | None = None,
    started: float,
    training_seconds: float,
) -> dict[str, Any]:
    probe, target, decoded = probe_data
    headline = probe["layers"][HEADLINE_LAYER]
    write_probe_metrics(context.results_dir / "probe_metrics.json", probe)
    plot_belief_comparison(
        target,
        decoded,
        path=context.results_dir / "belief_simplex_comparison.png",
        mse=headline["mse"],
        r2=headline["r2"],
        seed=setup.plot_seed,
    )
    plot_probe_trajectory(
        trajectory,
        bayesian_accuracy=setup.bayesian_accuracy,
        title=arm,
        path=context.results_dir / "probe_trajectory.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {"arm": arm, "checkpoints": trajectory},
    )
    applicable = not context.smoke
    checks = {"final_pre_ln_mse": headline["mse"] <= FINAL_MSE_THRESHOLD}
    checks.update(extra_checks)
    final = trajectory[-1]
    summary = {
        "arm": arm,
        "seed": context.seed,
        "smoke": context.smoke,
        "final_agent_steps": final["agent_steps"],
        "bayesian_floor_nats": setup.bayesian_floor,
        "bayesian_greedy_accuracy": setup.bayesian_accuracy,
        "final_policy": {
            key: final[key]
            for key in (
                "cross_entropy_nats",
                "greedy_accuracy",
                "expected_sampled_reward",
            )
        },
        "final_probe": probe,
        "initial_headline_r2": trajectory[0]["headline_r2"],
        "final_headline_r2": headline["r2"],
        "final_headline_mse": headline["mse"],
        **(extra_summary or {}),
        "success_checks": {
            "applicable": applicable,
            "final_pre_ln_mse_threshold": FINAL_MSE_THRESHOLD,
            "checks": checks,
            "passed": all(checks.values()) if applicable else None,
        },
        "timing": {
            "training_wall_seconds": training_seconds,
            "experiment_wall_seconds": time.monotonic() - started,
        },
        "checkpoint_probes": trajectory,
    }
    outputs.write_json("summary.json", summary)
    return summary


def run_paper_supervised(
    context: RunContext,
    *,
    full_config: TrainingConfig,
    smoke_config: TrainingConfig,
) -> dict[str, Any]:
    """Paper recipe: next-token cross-entropy with plain SGD on fresh data."""
    started = time.monotonic()
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    setup = prepare(context)
    config = smoke_config if context.smoke else full_config
    _seed_model(setup.initialization_seed)
    model = PaperTransformer(MODEL_CONFIG).to(setup.device)
    positions = MODEL_CONFIG.context_length
    outputs.write_json(
        "resolved_recipe.json",
        _recipe(
            context,
            setup,
            arm="paper_supervised",
            objective="ten shifted next-token cross-entropies",
            model=model,
            training={
                **config.to_dict(),
                "agent_steps_per_update": config.batch_size * positions,
            },
        ),
    )
    history, training_summary, _ = train(
        model=model,
        paths=setup.paths,
        probabilities=setup.probabilities,
        alias_table=setup.alias_table,
        device=setup.device,
        seed=setup.training_seed,
        config=config,
        outputs=outputs,
        resume_from=context.resume_from,
    )
    plot_training_curve(
        history,
        floor_nats=setup.bayesian_floor,
        path=context.results_dir / "training_validation_curve.png",
    )
    checkpoints = [
        (
            int(path.stem.removeprefix("step_")) * config.batch_size * positions,
            path,
        )
        for path in outputs.checkpoints_dir.glob("step_*.pt")
    ]

    def build_model() -> PaperTransformer:
        return PaperTransformer(MODEL_CONFIG).to(setup.device)

    trajectory, probe_data = probe_checkpoints(
        checkpoints,
        build_model=build_model,
        setup=setup,
        context=context,
        extra=lambda path: {"update": int(path.stem.removeprefix("step_"))},
    )
    validation_gap = trajectory[-1]["cross_entropy_nats"] - setup.bayesian_floor
    return _finish(
        context,
        outputs,
        setup,
        arm="paper_supervised",
        trajectory=trajectory,
        probe_data=probe_data,
        extra_checks={
            "validation_gap": validation_gap <= VALIDATION_GAP_THRESHOLD_NATS
        },
        extra_summary={"validation_gap_nats": validation_gap},
        started=started,
        training_seconds=training_summary["end_to_end_training_wall_seconds"],
    )


def run_ppo_token_guess(
    context: RunContext,
    *,
    full_config: PPOConfig,
    smoke_config: PPOConfig,
) -> dict[str, Any]:
    """Gamma-zero PPO rewarded only for correct next-token guesses."""
    started = time.monotonic()
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    setup = prepare(context)
    config = smoke_config if context.smoke else full_config
    positions = MODEL_CONFIG.context_length
    _seed_model(setup.initialization_seed)
    model = PaperActorCritic(MODEL_CONFIG).to(setup.device)
    outputs.write_json(
        "resolved_recipe.json",
        _recipe(
            context,
            setup,
            arm="ppo_token_guess",
            objective="clipped PPO, gamma 0, reward 1[guess == next token]",
            model=model,
            training={
                **config.to_dict(),
                "optimizer": "adam",
                "gamma": 0.0,
                "iterations": config.iterations(positions),
                "agent_steps_per_iteration": config.rollout_sequences * positions,
                "gradient_updates_per_iteration": config.num_epochs
                * (config.rollout_sequences // config.minibatch_sequences),
                "advantage": "reward - value (no normalization)",
                "uses_cross_entropy": False,
            },
        ),
    )
    training_started = time.monotonic()
    _, checkpoints = train_ppo(
        model=model,
        paths=setup.paths,
        alias_table=setup.alias_table,
        device=setup.device,
        seed=setup.training_seed,
        config=config,
        outputs=outputs,
        resume_from=context.resume_from,
    )
    training_seconds = time.monotonic() - training_started

    def build_model() -> PaperActorCritic:
        return PaperActorCritic(MODEL_CONFIG).to(setup.device)

    trajectory, probe_data = probe_checkpoints(
        checkpoints,
        build_model=build_model,
        setup=setup,
        context=context,
    )
    return _finish(
        context,
        outputs,
        setup,
        arm="ppo_token_guess",
        trajectory=trajectory,
        probe_data=probe_data,
        extra_checks={},
        started=started,
        training_seconds=training_seconds,
    )
