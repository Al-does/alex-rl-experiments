"""Pure next-token optimization with resumable device-native checkpoints."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from harness.artifacts import RunArtifacts

from .model import FactoredNextTokenTransformer
from .process import SEQUENCE_LENGTH, SequenceBatch, sample_sequences


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """The paper's optimizer and budget, plus compact checkpoint controls."""

    total_updates: int = 500_000
    batch_size: int = 25_000
    learning_rate: float = 5e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    sequence_length: int = SEQUENCE_LENGTH
    validation_batch_size: int = 4_096
    log_every: int = 1_000

    @classmethod
    def smoke(cls) -> TrainingConfig:
        return cls(
            total_updates=10,
            batch_size=24,
            validation_batch_size=128,
            log_every=1,
        )

    def __post_init__(self) -> None:
        if min(
            self.total_updates,
            self.batch_size,
            self.sequence_length,
            self.validation_batch_size,
            self.log_every,
        ) <= 0:
            raise ValueError("training counts must be positive")
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError("this paper replication fixes sequence length at eight")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_updates(total_updates: int) -> tuple[int, ...]:
    """Return initialization, powers of two, and the final update."""

    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    updates = {0, total_updates}
    update = 1
    while update < total_updates:
        updates.add(update)
        update *= 2
    return tuple(sorted(updates))


def language_model_io(
    batch: SequenceBatch,
    model: FactoredNextTokenTransformer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the paper's ``[BOS, x1, ..., x7] -> [x1, ..., x8]`` shift."""

    bos = torch.full(
        (len(batch.tokens), 1),
        model.config.bos_token,
        dtype=torch.long,
        device=batch.tokens.device,
    )
    inputs = torch.cat([bos, batch.tokens[:, :-1]], dim=1)
    return inputs, batch.tokens


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.detach().cpu()
    return state


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    *,
    model: FactoredNextTokenTransformer,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator | None,
    update: int,
    history: list[dict[str, Any]],
    config: TrainingConfig,
) -> None:
    _atomic_torch_save(
        {
            "update": update,
            "model_config": model.config.to_dict(),
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "optimizer_state": _optimizer_state_to_cpu(optimizer),
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
    model: FactoredNextTokenTransformer,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = model.config.to_dict()
    recorded = checkpoint.get("model_config", {})
    if recorded.get("factor_count") != expected["factor_count"]:
        raise ValueError("checkpoint factor count does not match the model")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if generator is not None and checkpoint.get("generator_state") is not None:
        generator.set_state(checkpoint["generator_state"].cpu())
    elif checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    return int(checkpoint["update"]), list(checkpoint.get("history", []))


@torch.inference_mode()
def evaluate(
    model: FactoredNextTokenTransformer,
    batch: SequenceBatch,
) -> dict[str, float]:
    """Evaluate held-out next-token loss, accuracy, and the Bayesian floor."""

    was_training = model.training
    model.eval()
    inputs, targets = language_model_io(batch, model)
    logits = model.forward(inputs)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    bayes_loss = -batch.target_probabilities.log().mean()
    if was_training:
        model.train()
    return {
        "validation_loss_nats": float(loss.cpu()),
        "validation_accuracy": float(accuracy.cpu()),
        "bayes_loss_nats": float(bayes_loss.cpu()),
        "validation_gap_nats": float((loss - bayes_loss).cpu()),
    }


def train(
    *,
    model: FactoredNextTokenTransformer,
    factor_count: int,
    device: torch.device,
    seed: int,
    validation_seed: int,
    config: TrainingConfig,
    outputs: RunArtifacts,
    resume_from: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    """Optimize only shifted next-token cross entropy."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta_1, config.beta_2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(validation_seed)
    except RuntimeError:
        generator = None
        validation_generator = None
        torch.manual_seed(seed)

    validation_batch = sample_sequences(
        batch_size=config.validation_batch_size,
        factor_count=factor_count,
        sequence_length=config.sequence_length,
        device=device,
        generator=validation_generator,
    )
    checkpoints = outputs.checkpoints_dir
    checkpoints.mkdir(parents=True, exist_ok=True)
    start_update = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        resume_path = (
            resume_from / "latest.pt" if resume_from.is_dir() else resume_from
        )
        start_update, history = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            generator=generator,
            device=device,
        )
        if start_update > config.total_updates:
            raise ValueError("checkpoint exceeds the configured update budget")

    milestones = set(checkpoint_updates(config.total_updates))
    retained: list[Path] = []
    started_at = time.monotonic()
    running_loss = torch.zeros((), device=device)
    running_updates = 0

    def record(update: int, training_loss: float | None) -> dict[str, Any]:
        validation = evaluate(model, validation_batch)
        row = {
            "update": update,
            "sequences_seen": update * config.batch_size,
            "target_tokens_seen": (
                update * config.batch_size * config.sequence_length
            ),
            "training_loss_nats": training_loss,
            **validation,
            "wall_seconds": time.monotonic() - started_at,
        }
        history.append(row)
        outputs.append_result(row)
        outputs.append_jsonl("training_curve.jsonl", row)
        return row

    if start_update == 0:
        record(0, None)
        initial = checkpoints / "update_000000.pt"
        save_checkpoint(
            initial,
            model=model,
            optimizer=optimizer,
            generator=generator,
            update=0,
            history=history,
            config=config,
        )
        retained.append(initial)

    compiled_training = device.type == "cuda"
    if compiled_training:
        model.compile(mode="reduce-overhead", fullgraph=True)
    model.train()
    for update in range(start_update + 1, config.total_updates + 1):
        batch = sample_sequences(
            batch_size=config.batch_size,
            factor_count=factor_count,
            sequence_length=config.sequence_length,
            device=device,
            generator=generator,
        )
        inputs, targets = language_model_io(batch, model)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.detach()
        running_updates += 1

        log_due = update % config.log_every == 0
        checkpoint_due = update in milestones
        if not (log_due or checkpoint_due):
            continue
        _synchronize(device)
        mean_loss = float((running_loss / running_updates).cpu())
        if log_due or checkpoint_due:
            record(update, mean_loss)
        running_loss.zero_()
        running_updates = 0

        if checkpoint_due:
            path = checkpoints / f"update_{update:06d}.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                generator=generator,
                update=update,
                history=history,
                config=config,
            )
            save_checkpoint(
                checkpoints / "latest.pt",
                model=model,
                optimizer=optimizer,
                generator=generator,
                update=update,
                history=history,
                config=config,
            )
            retained.append(path)

    final = next(row for row in reversed(history) if row["update"] == config.total_updates)
    summary = {
        "start_update": start_update,
        "completed_update": config.total_updates,
        "sequences_seen": config.total_updates * config.batch_size,
        "target_tokens_seen": (
            config.total_updates * config.batch_size * config.sequence_length
        ),
        "compiled_training": compiled_training,
        "final_validation_loss_nats": final["validation_loss_nats"],
        "final_validation_accuracy": final["validation_accuracy"],
        "bayes_loss_nats": final["bayes_loss_nats"],
        "final_validation_gap_nats": final["validation_gap_nats"],
        "training_wall_seconds": time.monotonic() - started_at,
    }
    return history, summary, retained
