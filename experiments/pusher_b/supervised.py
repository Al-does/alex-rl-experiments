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
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from .process import (
    BOS_TOKEN,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    PRESETS,
    STATE_COUNT,
    TOKEN_COUNT,
    pusher_b_model,
)
from .supervised_model import ModelConfig, PusherBTransformer, parameter_count


CHECKPOINT_STEPS = (1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000)
MODEL_CONFIG = ModelConfig()
_STREAM_KEYS = {
    "model_initialization": (0,),
    "training_sampling": (1,),
    "validation_sampling": (2,),
}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    total_steps: int = 10_000
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS
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
        if self.batch_size <= 0 or self.validation_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint_steps must be unique and sorted")
        if not self.checkpoint_steps:
            raise ValueError("at least one checkpoint step is required")
        if self.checkpoint_steps[-1] != self.total_steps:
            raise ValueError("the final checkpoint must equal total_steps")

    @classmethod
    def smoke(cls) -> TrainingConfig:
        return cls(
            total_steps=2,
            checkpoint_steps=(1, 2),
            batch_size=2,
            log_every=1,
            validation_batch_size=4,
            validation_minibatch_size=2,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SequenceSampler:
    def __init__(self, *, preset: str, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        model = pusher_b_model(preset)
        self.initial_distribution = np.asarray(
            model.initial_distribution,
            dtype=np.float64,
        )
        edges = np.asarray(model.edge_transition_matrices, dtype=np.float64)
        flattened = edges.transpose(1, 0, 2).reshape(
            STATE_COUNT,
            TOKEN_COUNT * STATE_COUNT,
        )
        self._edge_cdf = np.cumsum(flattened, axis=1)
        self._edge_cdf[:, -1] = 1.0

    @property
    def rng_state(self) -> dict[str, Any]:
        return self._rng.bit_generator.state

    @rng_state.setter
    def rng_state(self, value: dict[str, Any]) -> None:
        self._rng.bit_generator.state = value

    def sample(
        self,
        batch_size: int,
        *,
        emission_count: int = EPISODE_LENGTH,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if emission_count <= 0:
            raise ValueError("emission_count must be positive")
        if emission_count + 1 > CONTEXT_LENGTH:
            raise ValueError("sequence exceeds configured context length")
        states = np.empty((batch_size, emission_count + 1), dtype=np.int64)
        tokens = np.full(
            (batch_size, emission_count + 1),
            BOS_TOKEN,
            dtype=np.int64,
        )
        states[:, 0] = self._rng.choice(
            STATE_COUNT,
            size=batch_size,
            p=self.initial_distribution,
        )
        for position in range(1, emission_count + 1):
            current = states[:, position - 1]
            choices = (
                self._rng.random(batch_size)[:, None]
                > self._edge_cdf[current]
            ).sum(axis=1)
            tokens[:, position] = choices // STATE_COUNT
            states[:, position] = choices % STATE_COUNT
        return tokens


def beliefs_from_tokens(tokens: np.ndarray, *, preset: str) -> np.ndarray:
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if tokens.shape[1] > CONTEXT_LENGTH:
        raise ValueError("sequence exceeds configured context length")
    if not np.all(tokens[:, 0] == BOS_TOKEN):
        raise ValueError("every sequence must begin with BOS")
    if tokens.shape[1] > 1 and (
        (tokens[:, 1:] < 0).any() or (tokens[:, 1:] >= TOKEN_COUNT).any()
    ):
        raise ValueError("emissions must be token indices zero or one")
    model = pusher_b_model(preset)
    edges = np.asarray(model.edge_transition_matrices, dtype=np.float64)
    belief = np.broadcast_to(
        np.asarray(model.initial_distribution, dtype=np.float64),
        (len(tokens), STATE_COUNT),
    ).copy()
    result = np.empty((*tokens.shape, STATE_COUNT), dtype=np.float64)
    result[:, 0] = belief
    for position in range(1, tokens.shape[1]):
        operators = edges[tokens[:, position]]
        belief = np.einsum("bi,bij->bj", belief, operators)
        belief /= belief.sum(axis=1, keepdims=True)
        result[:, position] = belief
    return result


def next_token_distributions(
    beliefs: np.ndarray,
    *,
    preset: str,
) -> np.ndarray:
    beliefs = np.asarray(beliefs, dtype=np.float64)
    if beliefs.shape[-1] != STATE_COUNT:
        raise ValueError("belief width must equal the three Pusher-B states")
    emission = np.asarray(
        pusher_b_model(preset).emission_matrix,
        dtype=np.float64,
    )
    return beliefs @ emission


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


def _save_checkpoint(
    path: Path,
    *,
    model: PusherBTransformer,
    optimizer: torch.optim.Optimizer,
    sampler: SequenceSampler,
    step: int,
    history: list[dict[str, Any]],
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
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
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: PusherBTransformer,
    optimizer: torch.optim.Optimizer,
    sampler: SequenceSampler,
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
    model: PusherBTransformer,
    tokens: np.ndarray,
    exact_distributions: np.ndarray,
    *,
    device: torch.device,
    minibatch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    model_loss_sum = 0.0
    floor_loss_sum = 0.0
    token_count = 0
    for start in range(0, len(tokens), minibatch_size):
        stop = start + minibatch_size
        token_batch = torch.as_tensor(
            tokens[start:stop],
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
    preset: str,
    model: PusherBTransformer,
    device: torch.device,
    training_seed: int,
    validation_seed: int,
    config: TrainingConfig,
    outputs: RunArtifacts,
    resume_from: Path | None,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    sampler = SequenceSampler(preset=preset, seed=training_seed)
    validation_tokens = SequenceSampler(
        preset=preset,
        seed=validation_seed,
    ).sample(config.validation_batch_size)
    validation_distributions = next_token_distributions(
        beliefs_from_tokens(validation_tokens, preset=preset),
        preset=preset,
    )
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
        start_step, history = _load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            device=device,
        )
        if start_step > config.total_steps:
            raise ValueError("checkpoint exceeds the configured training budget")

    started_at = time.monotonic()
    active_seconds = 0.0
    active_steps = 0
    running_loss = torch.zeros((), device=device)
    running_count = 0

    def evaluate(step: int) -> dict[str, Any]:
        metrics = evaluate_next_token_loss(
            model,
            validation_tokens,
            validation_distributions,
            device=device,
            minibatch_size=config.validation_minibatch_size,
        )
        record = {
            "kind": "validation",
            "step": step,
            **metrics,
            "end_to_end_wall_seconds": time.monotonic() - started_at,
        }
        history.append(record)
        outputs.append_result(record)
        return record

    if start_step == 0:
        evaluate(0)
        _save_checkpoint(
            checkpoints / "step_0000000.pt",
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            step=0,
            history=history,
            config=config,
        )

    compiled_forward = (
        torch.compile(model, mode="reduce-overhead", fullgraph=True)
        if device.type == "cuda"
        else model
    )
    model.train()
    for step in range(start_step + 1, config.total_steps + 1):
        tokens = torch.as_tensor(
            sampler.sample(config.batch_size),
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

        if step % config.log_every == 0 or step == config.total_steps:
            rate = active_steps / max(active_seconds, 1e-9)
            record = {
                "kind": "training",
                "step": step,
                "training_loss_nats": float(
                    (running_loss / running_count).cpu()
                ),
                "updates_per_second_active": rate,
                "sequences_per_second_active": rate * config.batch_size,
                "end_to_end_wall_seconds": time.monotonic() - started_at,
            }
            history.append(record)
            outputs.append_result(record)
            running_loss.zero_()
            running_count = 0

        if step in config.checkpoint_steps:
            evaluate(step)
            checkpoint = checkpoints / f"step_{step:07d}.pt"
            _save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                sampler=sampler,
                step=step,
                history=history,
                config=config,
            )
            _save_checkpoint(
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
        and record["step"] == config.total_steps
    )
    rate = active_steps / max(active_seconds, 1e-9)
    return {
        "start_step": start_step,
        "completed_step": config.total_steps,
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
        "end_to_end_training_wall_seconds": time.monotonic() - started_at,
    }


def run_supervised(context: RunContext, *, preset: str) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    if context.seed is None:
        raise ValueError("Pusher-B supervised training requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    device = _device(context)
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    seeds = {
        name: seed_sequence_to_int(stream, bits=64)
        for name, stream in streams.items()
    }
    training_config = (
        TrainingConfig.smoke() if context.smoke else TrainingConfig()
    )
    _seed_torch(seeds["model_initialization"])
    model = PusherBTransformer(MODEL_CONFIG).to(device)
    hmm = pusher_b_model(preset)
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "pusher_b",
            "condition": "supervised_next_token_prediction",
            "preset": preset,
            "parameters": PRESETS[preset],
            "source_guide": "https://github.com/Al-does/alex-rl-experiments/pull/130",
            "hmm": {
                "states": list(hmm.state_labels),
                "tokens": list(hmm.token_labels),
                "initial_distribution": hmm.initial_distribution.tolist(),
                "transition_matrix": hmm.transition_matrix.tolist(),
                "emission_matrix": hmm.emission_matrix.tolist(),
                "edge_transition_matrices": (
                    hmm.edge_transition_matrices.tolist()
                ),
            },
            "sequence": {
                "bos_token": BOS_TOKEN,
                "emissions": EPISODE_LENGTH,
                "context_length": CONTEXT_LENGTH,
            },
            "model": {
                **MODEL_CONFIG.to_dict(),
                "parameter_count": parameter_count(model),
            },
            "objective": "shifted next-token cross-entropy",
            "training": training_config.to_dict(),
            "runtime": {
                "seed": context.seed,
                "derived_seeds": seeds,
                "device": str(device),
                "smoke": context.smoke,
            },
        },
    )
    training = train(
        preset=preset,
        model=model,
        device=device,
        training_seed=seeds["training_sampling"],
        validation_seed=seeds["validation_sampling"],
        config=training_config,
        outputs=outputs,
        resume_from=context.resume_from,
    )
    summary = {
        "study": "pusher_b",
        "condition": "supervised_next_token_prediction",
        "preset": preset,
        "parameters": PRESETS[preset],
        "seed": context.seed,
        "smoke": context.smoke,
        "training": training,
    }
    outputs.write_json("summary.json", summary)
    (context.results_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Pusher-B supervised ({preset})",
                "",
                f"- Completed updates: {training['completed_step']:,}",
                (
                    "- Held-out loss gap: "
                    f"{training['excess_loss_nats']:.6g} nats/token"
                ),
                "",
            ]
        )
    )
    return summary
