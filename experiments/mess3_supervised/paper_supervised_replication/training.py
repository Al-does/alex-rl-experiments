"""Paper-faithful SGD loop with exact validation and resumable checkpoints."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from harness.artifacts import RunArtifacts
from learners.optimizer import partition_muon_params

from .mess3 import AliasTable
from .model import PaperTransformer


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    total_steps: int = 1_000_000
    analyzed_step: int = 983_140
    batch_size: int = 64
    optimizer_name: str = "sgd"
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.0
    auxiliary_learning_rate: float | None = None
    auxiliary_weight_decay: float = 0.0
    log_every: int = 1_000
    checkpoint_every: int = 10_000
    retain_periodic_checkpoints: bool = False
    validation_every: int = 50_000
    validation_batch_size: int = 4_096

    @classmethod
    def smoke(cls) -> "TrainingConfig":
        return cls(
            total_steps=100,
            analyzed_step=100,
            log_every=10,
            checkpoint_every=50,
            validation_every=50,
            validation_batch_size=1_024,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> dict:
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.detach().cpu()
    return state


def _build_optimizers(
    model: PaperTransformer,
    config: TrainingConfig,
) -> tuple[torch.optim.Optimizer, ...]:
    if config.optimizer_name == "sgd":
        return (
            torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
            ),
        )
    if config.optimizer_name == "muon":
        muon_params, auxiliary_params = partition_muon_params(model.parameters())
        if not muon_params or not auxiliary_params:
            raise ValueError(
                "Muon training requires both 2D and non-2D model parameters"
            )
        auxiliary_lr = (
            config.learning_rate
            if config.auxiliary_learning_rate is None
            else config.auxiliary_learning_rate
        )
        return (
            torch.optim.Muon(
                muon_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
            ),
            torch.optim.AdamW(
                auxiliary_params,
                lr=auxiliary_lr,
                weight_decay=config.auxiliary_weight_decay,
            ),
        )
    raise ValueError(f"unsupported optimizer: {config.optimizer_name!r}")


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    *,
    model: PaperTransformer,
    optimizers: tuple[torch.optim.Optimizer, ...],
    generator: torch.Generator | None,
    step: int,
    history: list[dict[str, Any]],
    config: TrainingConfig,
) -> None:
    _atomic_torch_save(
        {
            "step": step,
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "optimizer_states": [
                _optimizer_state_to_cpu(optimizer)
                for optimizer in optimizers
            ],
            "generator_state": (
                generator.get_state() if generator is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "history": history,
            "training_config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: PaperTransformer,
    optimizers: tuple[torch.optim.Optimizer, ...] | None,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizers is not None:
        optimizer_states = checkpoint.get("optimizer_states")
        if optimizer_states is None and checkpoint.get("optimizer_state") is not None:
            optimizer_states = [checkpoint["optimizer_state"]]
        if optimizer_states is None or len(optimizer_states) != len(optimizers):
            raise ValueError("checkpoint optimizer state does not match recipe")
        for optimizer, optimizer_state in zip(optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)
    if generator is not None and checkpoint.get("generator_state") is not None:
        generator.set_state(checkpoint["generator_state"].cpu())
    elif checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    return int(checkpoint["step"]), list(checkpoint.get("history", []))


@torch.no_grad()
def exact_validation_loss(
    model: PaperTransformer,
    paths: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    """Evaluate expected ten-position CE over all supplied length-11 paths."""
    was_training = model.training
    model.eval()
    total = torch.zeros(
        (),
        dtype=probabilities.dtype,
        device=paths.device,
    )
    normalization = probabilities.sum()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        # Bypass the compiled __call__ used by fixed-shape optimization. Exact
        # validation has different batch shapes and should stay eager.
        logits = model.forward(batch_paths[:, :-1])
        targets = batch_paths[:, 1:]
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(len(batch_paths), -1)
        per_path = per_token.mean(dim=-1).to(probabilities.dtype)
        total += (
            probabilities[start : start + len(batch_paths)] * per_path
        ).sum()
    if was_training:
        model.train()
    return float((total / normalization).cpu())


def validation_steps(config: TrainingConfig) -> set[int]:
    steps = {
        0,
        min(1_000, config.total_steps),
        min(10_000, config.total_steps),
        config.analyzed_step,
        config.total_steps,
    }
    steps.update(
        range(
            config.validation_every,
            config.total_steps + 1,
            config.validation_every,
        )
    )
    return {step for step in steps if 0 <= step <= config.total_steps}


def train(
    *,
    model: PaperTransformer,
    paths: torch.Tensor,
    probabilities: torch.Tensor,
    alias_table: AliasTable,
    device: torch.device,
    seed: int,
    config: TrainingConfig,
    outputs: RunArtifacts,
    resume_from: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Train with only device-native sampling/forward/loss ops in the hot loop."""
    optimizers = _build_optimizers(model, config)
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    except RuntimeError:
        generator = None
        torch.manual_seed(seed)

    checkpoints = outputs.checkpoints_dir
    checkpoints.mkdir(parents=True, exist_ok=True)
    start_step = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        resume_path = (
            resume_from / "latest.pt"
            if resume_from.is_dir()
            else resume_from
        )
        start_step, history = load_checkpoint(
            resume_path,
            model=model,
            optimizers=optimizers,
            generator=generator,
            device=device,
        )
        if start_step > config.total_steps:
            raise ValueError("checkpoint is beyond the configured training budget")

    evaluation_steps = validation_steps(config)
    started_at = time.monotonic()
    running_loss = torch.zeros((), device=device)
    running_count = 0
    active_update_seconds = 0.0
    active_update_count = 0
    segment_update_count = 0
    segment_started = time.monotonic()

    def evaluate(step: int) -> float:
        validation_started = time.monotonic()
        value = exact_validation_loss(
            model,
            paths,
            probabilities,
            batch_size=config.validation_batch_size,
        )
        record = {
            "kind": "validation",
            "step": step,
            "validation_loss_nats": value,
            "validation_wall_seconds": (
                time.monotonic() - validation_started
            ),
            "end_to_end_wall_seconds": time.monotonic() - started_at,
        }
        history.append(record)
        outputs.append_result(record)
        return value

    if start_step == 0:
        evaluate(0)
        save_checkpoint(
            checkpoints / "step_0000000.pt",
            model=model,
            optimizers=optimizers,
            generator=generator,
            step=0,
            history=history,
            config=config,
        )

    compiled_training = device.type == "cuda"
    if compiled_training:
        model.compile(mode="reduce-overhead", fullgraph=True)
    model.train()
    segment_started = time.monotonic()
    for step in range(start_step + 1, config.total_steps + 1):
        indices = alias_table.sample(
            config.batch_size,
            generator=generator,
        )
        sampled_paths = paths.index_select(0, indices)
        logits = model(sampled_paths[:, :-1])
        targets = sampled_paths[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        running_loss = running_loss + loss.detach()
        running_count += 1
        segment_update_count += 1
        log_due = step % config.log_every == 0 or step == config.total_steps
        validation_due = step in evaluation_steps
        milestone = step in {config.analyzed_step, config.total_steps}
        checkpoint_due = (
            step % config.checkpoint_every == 0 or milestone
        )
        maintenance_due = log_due or validation_due or checkpoint_due
        if maintenance_due:
            _synchronize(device)
            active_update_seconds += time.monotonic() - segment_started
            active_update_count += segment_update_count
            segment_update_count = 0

        if log_due:
            end_to_end_elapsed = time.monotonic() - started_at
            active_rate = active_update_count / max(
                active_update_seconds,
                1e-9,
            )
            record = {
                "kind": "training",
                "step": step,
                "training_loss_nats": float(
                    (running_loss / running_count).cpu()
                ),
                "active_optimization_wall_seconds": active_update_seconds,
                "updates_per_second_active": active_rate,
                "sequences_per_second_active": (
                    active_rate * config.batch_size
                ),
                "target_tokens_per_second_active": (
                    active_rate * config.batch_size * 10
                ),
                "end_to_end_wall_seconds": end_to_end_elapsed,
            }
            history.append(record)
            outputs.append_result(record)
            running_loss.zero_()
            running_count = 0

        if validation_due:
            evaluate(step)

        if checkpoint_due:
            save_checkpoint(
                checkpoints / "latest.pt",
                model=model,
                optimizers=optimizers,
                generator=generator,
                step=step,
                history=history,
                config=config,
            )
        if milestone or (
            config.retain_periodic_checkpoints
            and step % config.checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoints / f"step_{step:07d}.pt",
                model=model,
                optimizers=optimizers,
                generator=generator,
                step=step,
                history=history,
                config=config,
            )

        if maintenance_due:
            segment_started = time.monotonic()

    analyzed_path = checkpoints / f"step_{config.analyzed_step:07d}.pt"
    if not analyzed_path.exists():
        analyzed_path = checkpoints / f"step_{config.total_steps:07d}.pt"
    validation_records = [
        record for record in history if record["kind"] == "validation"
    ]
    final_validation = next(
        record["validation_loss_nats"]
        for record in reversed(validation_records)
        if record["step"] == config.total_steps
    )
    analyzed_validation = next(
        record["validation_loss_nats"]
        for record in reversed(validation_records)
        if record["step"] == config.analyzed_step
    )
    end_to_end_wall_seconds = time.monotonic() - started_at
    active_rate = active_update_count / max(active_update_seconds, 1e-9)
    summary = {
        "start_step": start_step,
        "completed_step": config.total_steps,
        "analyzed_step": config.analyzed_step,
        "compiled_training": compiled_training,
        "analyzed_checkpoint": str(analyzed_path),
        "analyzed_validation_loss_nats": analyzed_validation,
        "final_validation_loss_nats": final_validation,
        "active_optimization_wall_seconds": active_update_seconds,
        "updates_per_second_active": active_rate,
        "sequences_per_second_active": active_rate * config.batch_size,
        "target_tokens_per_second_active": (
            active_rate * config.batch_size * 10
        ),
        "end_to_end_training_wall_seconds": end_to_end_wall_seconds,
        # Compatibility name used by the experiment summary and older readers.
        "training_wall_seconds": end_to_end_wall_seconds,
    }
    return history, summary, analyzed_path
