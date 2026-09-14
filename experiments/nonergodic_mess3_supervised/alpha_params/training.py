"""AdamW next-token training with article-aligned checkpoints."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from harness.artifacts import RunArtifacts

from .data import (
    EPISODE_LENGTH,
    NonergodicSequenceSampler,
    next_token_distributions,
    weighted_beliefs,
)
from .model import NonergodicMess3Transformer


FIGURE_CHECKPOINT_STEPS = (
    1,
    3,
    10,
    30,
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    45_000,
)
CONVERGENCE_CHECKPOINT_STEPS = tuple(
    step for step in FIGURE_CHECKPOINT_STEPS if step <= 10_000
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    total_steps: int
    analyzed_step: int
    checkpoint_steps: tuple[int, ...]
    batch_size: int = 512
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    log_every: int = 100
    validation_batch_size: int = 4_096
    validation_minibatch_size: int = 512

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 1 <= self.analyzed_step <= self.total_steps:
            raise ValueError("analyzed_step must lie inside the training budget")
        if self.batch_size <= 0 or self.validation_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if not self.checkpoint_steps:
            raise ValueError("at least one checkpoint step is required")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint_steps must be unique and sorted")
        if self.checkpoint_steps[-1] != self.analyzed_step:
            raise ValueError("the final checkpoint must be the analyzed step")

    @classmethod
    def convergence(cls) -> TrainingConfig:
        return cls(
            total_steps=10_000,
            analyzed_step=10_000,
            checkpoint_steps=CONVERGENCE_CHECKPOINT_STEPS,
        )

    @classmethod
    def figure_checkpoint(cls) -> TrainingConfig:
        return cls(
            total_steps=45_000,
            analyzed_step=45_000,
            checkpoint_steps=FIGURE_CHECKPOINT_STEPS,
        )

    @classmethod
    def smoke(cls) -> TrainingConfig:
        return cls(
            total_steps=2,
            analyzed_step=2,
            checkpoint_steps=(1, 2),
            batch_size=2,
            log_every=1,
            validation_batch_size=4,
            validation_minibatch_size=2,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _optimizer_state_to_cpu(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.detach().cpu()
    return state


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_analysis_checkpoint(
    path: Path,
    *,
    model: NonergodicMess3Transformer,
    step: int,
) -> None:
    _atomic_save(
        {
            "step": step,
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        },
        path,
    )


def save_resume_checkpoint(
    path: Path,
    *,
    model: NonergodicMess3Transformer,
    optimizer: torch.optim.Optimizer,
    sampler: NonergodicSequenceSampler,
    step: int,
    history: list[dict[str, Any]],
    config: TrainingConfig,
) -> None:
    _atomic_save(
        {
            "step": step,
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "optimizer_state": _optimizer_state_to_cpu(optimizer),
            "sampler_rng_state": sampler.rng_state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "history": history,
            "training_config": config.to_dict(),
        },
        path,
    )


def load_analysis_checkpoint(
    path: Path,
    *,
    model: NonergodicMess3Transformer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return int(checkpoint["step"])


def load_resume_checkpoint(
    path: Path,
    *,
    model: NonergodicMess3Transformer,
    optimizer: torch.optim.Optimizer,
    sampler: NonergodicSequenceSampler,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    sampler.rng_state = checkpoint["sampler_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return int(checkpoint["step"]), list(checkpoint.get("history", []))


@torch.no_grad()
def evaluate_next_token_loss(
    model: NonergodicMess3Transformer,
    tokens: np.ndarray,
    exact_distributions: np.ndarray,
    *,
    device: torch.device,
    minibatch_size: int,
) -> dict[str, float]:
    """Evaluate model CE and the Bayesian floor on identical held-out tokens."""

    was_training = model.training
    model.eval()
    model_loss_sum = 0.0
    floor_loss_sum = 0.0
    token_count = 0
    for start in range(0, len(tokens), minibatch_size):
        token_batch = torch.as_tensor(
            tokens[start : start + minibatch_size],
            dtype=torch.long,
            device=device,
        )
        logits = model(token_batch)
        targets = token_batch[:, 1:]
        model_loss_sum += float(
            F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            ).cpu()
        )
        probabilities = np.take_along_axis(
            exact_distributions[start : start + len(token_batch), :-1],
            tokens[start : start + len(token_batch), 1:, None],
            axis=-1,
        )
        floor_loss_sum += float(-np.log(probabilities).sum())
        token_count += targets.numel()
    if was_training:
        model.train()
    model_loss = model_loss_sum / token_count
    floor_loss = floor_loss_sum / token_count
    return {
        "validation_loss_nats": model_loss,
        "bayesian_floor_nats": floor_loss,
        "excess_loss_nats": model_loss - floor_loss,
    }


def train(
    *,
    model: NonergodicMess3Transformer,
    device: torch.device,
    training_seed: int,
    validation_seed: int,
    config: TrainingConfig,
    outputs: RunArtifacts,
    resume_from: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    sampler = NonergodicSequenceSampler(training_seed)
    validation_batch = NonergodicSequenceSampler(validation_seed).sample(
        config.validation_batch_size
    )
    validation_beliefs = weighted_beliefs(validation_batch.tokens)
    validation_distributions = next_token_distributions(validation_beliefs)
    del validation_beliefs

    checkpoints = outputs.checkpoints_dir
    checkpoints.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    start_step = 0
    if resume_from is not None:
        resume_path = (
            resume_from / "latest.pt"
            if resume_from.is_dir()
            else resume_from
        )
        start_step, history = load_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            device=device,
        )
        if start_step > config.total_steps:
            raise ValueError("checkpoint exceeds the configured training budget")

    checkpoint_paths: list[Path] = []
    zero_checkpoint = checkpoints / "analysis_step_0000000.pt"
    if start_step == 0:
        save_analysis_checkpoint(zero_checkpoint, model=model, step=0)
    if zero_checkpoint.exists():
        checkpoint_paths.append(zero_checkpoint)

    validation_steps = {0, *config.checkpoint_steps}
    started_at = time.monotonic()
    active_seconds = 0.0
    active_steps = 0
    running_loss = torch.zeros((), device=device)
    running_count = 0

    def evaluate(step: int) -> dict[str, Any]:
        evaluation_started = time.monotonic()
        metrics = evaluate_next_token_loss(
            model,
            validation_batch.tokens,
            validation_distributions,
            device=device,
            minibatch_size=config.validation_minibatch_size,
        )
        record = {
            "kind": "validation",
            "step": step,
            **metrics,
            "validation_wall_seconds": time.monotonic() - evaluation_started,
            "end_to_end_wall_seconds": time.monotonic() - started_at,
        }
        history.append(record)
        outputs.append_result(record)
        return record

    if start_step == 0:
        evaluate(0)

    compiled_forward = (
        torch.compile(model, mode="reduce-overhead", fullgraph=True)
        if device.type == "cuda"
        else model
    )
    model.train()
    for step in range(start_step + 1, config.total_steps + 1):
        sampled = sampler.sample(
            config.batch_size,
            emission_count=EPISODE_LENGTH,
        )
        tokens = torch.as_tensor(
            sampled.tokens,
            dtype=torch.long,
            device=device,
        )
        update_started = time.monotonic()
        logits = compiled_forward(tokens)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _synchronize(device)
        active_seconds += time.monotonic() - update_started
        active_steps += 1
        running_loss += loss.detach()
        running_count += 1

        log_due = step % config.log_every == 0 or step == config.total_steps
        checkpoint_due = step in config.checkpoint_steps
        if log_due:
            rate = active_steps / max(active_seconds, 1e-9)
            record = {
                "kind": "training",
                "step": step,
                "training_loss_nats": float(
                    (running_loss / running_count).cpu()
                ),
                "active_optimization_wall_seconds": active_seconds,
                "updates_per_second_active": rate,
                "sequences_per_second_active": rate * config.batch_size,
                "target_tokens_per_second_active": (
                    rate * config.batch_size * EPISODE_LENGTH
                ),
                "end_to_end_wall_seconds": time.monotonic() - started_at,
            }
            history.append(record)
            outputs.append_result(record)
            running_loss.zero_()
            running_count = 0

        if step in validation_steps:
            evaluate(step)

        if checkpoint_due:
            path = checkpoints / f"analysis_step_{step:07d}.pt"
            save_analysis_checkpoint(path, model=model, step=step)
            checkpoint_paths.append(path)
            save_resume_checkpoint(
                checkpoints / "latest.pt",
                model=model,
                optimizer=optimizer,
                sampler=sampler,
                step=step,
                history=history,
                config=config,
            )

    final_validation = next(
        record
        for record in reversed(history)
        if record["kind"] == "validation"
        and record["step"] == config.analyzed_step
    )
    rate = active_steps / max(active_seconds, 1e-9)
    summary = {
        "start_step": start_step,
        "completed_step": config.total_steps,
        "analyzed_step": config.analyzed_step,
        "compiled_training": device.type == "cuda",
        **{
            key: final_validation[key]
            for key in (
                "validation_loss_nats",
                "bayesian_floor_nats",
                "excess_loss_nats",
            )
        },
        "active_optimization_wall_seconds": active_seconds,
        "updates_per_second_active": rate,
        "sequences_per_second_active": rate * config.batch_size,
        "target_tokens_per_second_active": (
            rate * config.batch_size * EPISODE_LENGTH
        ),
        "end_to_end_training_wall_seconds": time.monotonic() - started_at,
    }
    return history, summary, checkpoint_paths
