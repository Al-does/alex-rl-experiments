"""Frozen-trunk transformer probes for exact next-token prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments.mess3_belief_geometry_2026_07.probe import ProbeData


N_ACTIONS = 3
N_TOKENS = 3


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    """Causal trunk windows paired with exact and sampled next-token targets."""

    embeddings: np.ndarray
    actions: np.ndarray
    target_probabilities: np.ndarray
    target_tokens: np.ndarray

    def __len__(self) -> int:
        return len(self.target_tokens)


@dataclass(frozen=True, slots=True)
class ProbeTrainingConfig:
    """Optimization settings shared by every checkpoint and condition."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    batch_size: int = 4096
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


def exact_next_token_probabilities(
    beliefs: np.ndarray,
    actions: np.ndarray,
    transition_matrices: np.ndarray,
    emission_matrix: np.ndarray,
) -> np.ndarray:
    """Return ``P(token[t+1] | history<=t, action[t])`` for delay zero."""

    beliefs = np.asarray(beliefs, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    transitions = np.asarray(transition_matrices, dtype=np.float64)
    emissions = np.asarray(emission_matrix, dtype=np.float64)
    if beliefs.ndim != 2:
        raise ValueError("beliefs must have shape (samples, states)")
    if len(actions) != len(beliefs):
        raise ValueError("actions and beliefs must have equal length")
    expected = (N_ACTIONS, beliefs.shape[1], beliefs.shape[1])
    if transitions.shape != expected:
        raise ValueError(
            f"transition_matrices must have shape {expected}, got "
            f"{transitions.shape}"
        )
    if emissions.shape != (beliefs.shape[1], N_TOKENS):
        raise ValueError("emission_matrix has incompatible shape")
    if ((actions < 0) | (actions >= N_ACTIONS)).any():
        raise ValueError("actions contain an invalid class")

    next_state = np.einsum(
        "bi,bij->bj",
        beliefs,
        transitions[actions],
        optimize=True,
    )
    probabilities = next_state @ emissions
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _sequence_indices(
    data: ProbeData,
    context_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build windows without crossing environment or episode boundaries."""

    if context_len <= 0:
        raise ValueError("context_len must be positive")
    windows: list[np.ndarray] = []
    next_indices: list[np.ndarray] = []
    offsets = np.arange(context_len, dtype=np.int64)
    for env_index in np.unique(data.env_indices):
        members = np.flatnonzero(data.env_indices == env_index)
        steps = data.episode_steps[members]
        boundaries = np.flatnonzero(np.diff(steps) != 1) + 1
        for segment in np.split(members, boundaries):
            if len(segment) <= context_len:
                continue
            starts = np.arange(
                0,
                len(segment) - context_len,
                dtype=np.int64,
            )
            windows.append(segment[starts[:, None] + offsets[None, :]])
            next_indices.append(segment[starts + context_len])
    if not windows:
        raise ValueError("rollout contains no complete next-token windows")
    return np.concatenate(windows), np.concatenate(next_indices)


def build_sequence_dataset(
    data: ProbeData,
    *,
    context_len: int,
    transition_matrices: np.ndarray,
    emission_matrix: np.ndarray,
) -> SequenceDataset:
    """Convert interleaved rollout rows into causal per-environment windows."""

    windows, next_indices = _sequence_indices(data, context_len)
    end_indices = windows[:, -1]
    actions = np.asarray(data.actions, dtype=np.int64).reshape(len(data.actions), -1)
    final_actions = actions[end_indices, 0]
    target_tokens = np.asarray(data.tokens, dtype=np.int64)[next_indices]
    if ((target_tokens < 0) | (target_tokens >= N_TOKENS)).any():
        raise ValueError("next-token targets contain an unobserved token")
    target_probabilities = exact_next_token_probabilities(
        data.beliefs[end_indices],
        final_actions,
        transition_matrices,
        emission_matrix,
    )
    return SequenceDataset(
        embeddings=np.asarray(data.activations[windows], dtype=np.float32),
        actions=final_actions,
        target_probabilities=target_probabilities,
        target_tokens=target_tokens,
    )


class NextTokenTransformerProbe(nn.Module):
    """A diagnostic transformer whose inputs are detached trunk embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        context_len: int,
        *,
        condition_on_action: bool,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.context_len = int(context_len)
        self.condition_on_action = bool(condition_on_action)
        self.input_projection = nn.Linear(embedding_dim + N_ACTIONS, d_model)
        self.positions = nn.Parameter(torch.empty(1, context_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, N_TOKENS)
        nn.init.normal_(self.positions, mean=0.0, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, context, width)")
        if embeddings.shape[1] != self.context_len:
            raise ValueError("embedding context length does not match probe")
        # The explicit detach is the stop-gradient boundary. The restored RL
        # trunk is also run under no_grad before these arrays are materialized.
        embeddings = embeddings.detach()
        action_features = embeddings.new_zeros(
            embeddings.shape[0],
            embeddings.shape[1],
            N_ACTIONS,
        )
        if self.condition_on_action:
            action_features[:, -1] = F.one_hot(
                actions.to(dtype=torch.long),
                num_classes=N_ACTIONS,
            ).to(dtype=embeddings.dtype)
        inputs = torch.cat([embeddings, action_features], dim=-1)
        hidden = self.input_projection(inputs) + self.positions
        hidden = self.transformer(hidden)
        return self.output(self.final_norm(hidden[:, -1]))


def _device_dataset(
    data: SequenceDataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(data.embeddings, device=device),
        torch.as_tensor(data.actions, dtype=torch.long, device=device),
        torch.as_tensor(data.target_probabilities, device=device),
        torch.as_tensor(data.target_tokens, dtype=torch.long, device=device),
    )


def _soft_cross_entropy(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    return -(probabilities * F.log_softmax(logits, dim=-1)).sum(dim=-1)


@torch.no_grad()
def _mean_soft_cross_entropy(
    model: NextTokenTransformerProbe,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
) -> float:
    embeddings, actions, probabilities, _ = data
    total = embeddings.new_zeros(())
    for start in range(0, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        total += _soft_cross_entropy(
            model(embeddings[start:stop], actions[start:stop]),
            probabilities[start:stop],
        ).sum()
    return float((total / len(embeddings)).item())


@torch.no_grad()
def evaluate_probe(
    model: NextTokenTransformerProbe,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    train_marginal: torch.Tensor,
    batch_size: int,
) -> dict[str, float | int]:
    """Evaluate predictive KL plus sampled-token calibration metrics."""

    embeddings, actions, probabilities, tokens = data
    totals = {
        name: embeddings.new_zeros(())
        for name in (
            "soft_cross_entropy",
            "bayes_entropy",
            "hard_cross_entropy",
            "bayes_hard_cross_entropy",
            "brier",
            "correct",
        )
    }
    for start in range(0, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        logits = model(embeddings[start:stop], actions[start:stop])
        log_probabilities = F.log_softmax(logits, dim=-1)
        predictions = log_probabilities.exp()
        targets = probabilities[start:stop]
        labels = tokens[start:stop]
        totals["soft_cross_entropy"] += (
            -(targets * log_probabilities).sum(dim=-1).sum()
        )
        totals["bayes_entropy"] += (
            -(targets * targets.clamp_min(1e-12).log()).sum(dim=-1).sum()
        )
        totals["hard_cross_entropy"] += (
            -log_probabilities.gather(1, labels[:, None]).sum()
        )
        totals["bayes_hard_cross_entropy"] += (
            -targets.clamp_min(1e-12).log().gather(1, labels[:, None]).sum()
        )
        totals["brier"] += ((predictions - targets) ** 2).sum(dim=-1).sum()
        totals["correct"] += (logits.argmax(dim=-1) == labels).sum()

    count = len(embeddings)
    values = {name: float((value / count).item()) for name, value in totals.items()}
    baseline_soft_ce = float(
        (
            -(probabilities * train_marginal.clamp_min(1e-12).log())
            .sum(dim=-1)
            .mean()
        ).item()
    )
    soft_kl = values["soft_cross_entropy"] - values["bayes_entropy"]
    baseline_kl = baseline_soft_ce - values["bayes_entropy"]
    return {
        "n_test": count,
        "soft_cross_entropy_nats": values["soft_cross_entropy"],
        "bayes_entropy_nats": values["bayes_entropy"],
        "soft_kl_nats": soft_kl,
        "marginal_baseline_cross_entropy_nats": baseline_soft_ce,
        "marginal_baseline_kl_nats": baseline_kl,
        "fraction_predictive_kl_removed": (
            1.0 - soft_kl / baseline_kl if baseline_kl > 0.0 else float("nan")
        ),
        "sampled_token_cross_entropy_nats": values["hard_cross_entropy"],
        "sampled_bayes_cross_entropy_nats": values[
            "bayes_hard_cross_entropy"
        ],
        "sampled_token_accuracy": values["correct"],
        "brier_score": values["brier"],
    }


def fit_probe(
    train: SequenceDataset,
    validation: SequenceDataset,
    test: SequenceDataset,
    *,
    condition_on_action: bool,
    device: str | torch.device,
    seed: int,
    config: ProbeTrainingConfig,
) -> dict[str, Any]:
    """Fit one deterministic probe and score its held-out test stream once."""

    if not train.embeddings.shape[2] == validation.embeddings.shape[2] == (
        test.embeddings.shape[2]
    ):
        raise ValueError("train, validation, and test widths must match")
    context_len = train.embeddings.shape[1]
    if validation.embeddings.shape[1] != context_len:
        raise ValueError("validation context length does not match training")
    if test.embeddings.shape[1] != context_len:
        raise ValueError("test context length does not match training")

    resolved_device = torch.device(device)
    torch.manual_seed(int(seed))
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = NextTokenTransformerProbe(
        train.embeddings.shape[2],
        context_len,
        condition_on_action=condition_on_action,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_tensors = _device_dataset(train, resolved_device)
    validation_tensors = _device_dataset(validation, resolved_device)
    test_tensors = _device_dataset(test, resolved_device)
    embeddings, actions, probabilities, _ = train_tensors
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed) + 1)
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_completed = 0

    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(
            len(embeddings),
            generator=generator,
            device=resolved_device,
        )
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = _soft_cross_entropy(
                model(embeddings[indices], actions[indices]),
                probabilities[indices],
            ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        validation_loss = _mean_soft_cross_entropy(
            model,
            validation_tensors,
            config.batch_size,
        )
        epochs_completed = epoch + 1
        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("probe optimization did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    train_marginal = train_tensors[2].mean(dim=0)
    metrics = evaluate_probe(
        model,
        test_tensors,
        train_marginal=train_marginal,
        batch_size=config.batch_size,
    )
    return {
        "probe": "frozen_trunk_transformer",
        "stop_gradient": True,
        "context_len": context_len,
        "condition_on_selected_action": condition_on_action,
        "target": "exact_next_visible_token_distribution",
        "loss": "soft_cross_entropy_equivalently_forward_kl_up_to_target_entropy",
        "epochs_completed": epochs_completed,
        "best_validation_cross_entropy_nats": best_validation,
        "n_train": len(train),
        "n_validation": len(validation),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **metrics,
    }
