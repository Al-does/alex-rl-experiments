"""Gamma-zero clipped PPO on fresh stationary MESS3 sequences.

Every sequence position is one agent step: the policy sees tokens up to that
position, guesses the next token, and receives reward 1 for a correct guess.
MESS3 is passive and the policy input contains only tokens, so one causal
forward pass over a sampled sequence is exactly equivalent to ten sequential
decisions. Every rollout samples new sequences; no rollout data is reused
across iterations.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from harness.artifacts import RunArtifacts

from .mess3 import AliasTable
from .model import PaperActorCritic


@dataclass(frozen=True, slots=True)
class PPOConfig:
    total_agent_steps: int = 2_000_000
    rollout_sequences: int = 4_000
    minibatch_sequences: int = 500
    num_epochs: int = 6
    learning_rate: float = 1e-4
    clip_param: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.0
    checkpoint_every_iterations: int = 10

    @classmethod
    def smoke(cls) -> "PPOConfig":
        return cls(
            total_agent_steps=4_000,
            rollout_sequences=100,
            minibatch_sequences=25,
            num_epochs=2,
            checkpoint_every_iterations=2,
        )

    def iterations(self, positions_per_sequence: int) -> int:
        steps_per_iteration = self.rollout_sequences * positions_per_sequence
        if self.total_agent_steps % steps_per_iteration:
            raise ValueError(
                "total_agent_steps must be a multiple of rollout agent steps"
            )
        if self.rollout_sequences % self.minibatch_sequences:
            raise ValueError(
                "rollout_sequences must be a multiple of minibatch_sequences"
            )
        return self.total_agent_steps // steps_per_iteration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _model_state(model: PaperActorCritic) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }


@torch.no_grad()
def collect_rollout(
    model: PaperActorCritic,
    paths: torch.Tensor,
    alias_table: AliasTable,
    *,
    sequences: int,
    generator: torch.Generator | None,
) -> dict[str, torch.Tensor]:
    """Sample fresh sequences, stochastic guesses, rewards, and baselines."""
    indices = alias_table.sample(sequences, generator=generator)
    sampled = paths.index_select(0, indices)
    contexts, targets = sampled[:, :-1], sampled[:, 1:]
    logits, values = model.policy_and_value(contexts)
    log_probabilities = F.log_softmax(logits, dim=-1)
    actions = torch.multinomial(
        log_probabilities.exp().reshape(-1, logits.shape[-1]),
        1,
        generator=generator,
    ).reshape(targets.shape)
    rewards = (actions == targets).to(values.dtype)
    return {
        "contexts": contexts,
        "targets": targets,
        "actions": actions,
        "old_log_probs": log_probabilities.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1),
        "rewards": rewards,
        # Gamma zero: the return is the immediate reward.
        "advantages": rewards - values,
    }


def ppo_loss(
    model: PaperActorCritic,
    batch: dict[str, torch.Tensor],
    config: PPOConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, values = model.policy_and_value(batch["contexts"])
    log_probabilities = F.log_softmax(logits, dim=-1)
    log_probs = log_probabilities.gather(
        -1, batch["actions"].unsqueeze(-1)
    ).squeeze(-1)
    ratio = torch.exp(log_probs - batch["old_log_probs"])
    advantages = batch["advantages"]
    clipped = ratio.clamp(1.0 - config.clip_param, 1.0 + config.clip_param)
    policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    value_loss = F.mse_loss(values, batch["rewards"])
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean()
    loss = (
        policy_loss
        + config.value_loss_coeff * value_loss
        - config.entropy_coeff * entropy
    )
    return loss, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "approx_kl": (batch["old_log_probs"] - log_probs).mean().detach(),
        "clip_fraction": (
            (ratio - 1.0).abs() > config.clip_param
        ).float().mean().detach(),
    }


def train_ppo(
    *,
    model: PaperActorCritic,
    paths: torch.Tensor,
    alias_table: AliasTable,
    device: torch.device,
    seed: int,
    config: PPOConfig,
    outputs: RunArtifacts,
    resume_from: Path | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, Path]]]:
    """Train and return history plus ``(agent_steps, checkpoint)`` pairs."""
    positions = paths.shape[1] - 1
    iterations = config.iterations(positions)
    steps_per_iteration = config.rollout_sequences * positions
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    except RuntimeError:
        generator = None
        torch.manual_seed(seed)
    checkpoints = outputs.checkpoints_dir
    history: list[dict[str, Any]] = []
    start_iteration = 0
    if resume_from is not None:
        resume_path = (
            resume_from / "latest.pt" if resume_from.is_dir() else resume_from
        )
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if generator is not None:
            generator.set_state(state["generator_state"].cpu())
        start_iteration = int(state["iteration"])
        history = list(state["history"])

    def retained(iteration: int) -> Path:
        return checkpoints / f"iteration_{iteration:05d}.pt"

    def save(iteration: int) -> None:
        agent_steps = iteration * steps_per_iteration
        _save(
            retained(iteration),
            {"agent_steps": agent_steps, "model_state": _model_state(model)},
        )
        _save(
            checkpoints / "latest.pt",
            {
                "iteration": iteration,
                "agent_steps": agent_steps,
                "model_state": _model_state(model),
                "optimizer_state": optimizer.state_dict(),
                "generator_state": (
                    generator.get_state() if generator is not None else None
                ),
                "history": history,
                "config": config.to_dict(),
            },
        )

    if start_iteration == 0:
        save(0)
    started = time.monotonic()
    model.train()
    for iteration in range(start_iteration + 1, iterations + 1):
        rollout = collect_rollout(
            model,
            paths,
            alias_table,
            sequences=config.rollout_sequences,
            generator=generator,
        )
        totals: dict[str, torch.Tensor] = {}
        updates = 0
        for _ in range(config.num_epochs):
            order = torch.randperm(
                config.rollout_sequences, device=device, generator=generator
            )
            for start in range(
                0, config.rollout_sequences, config.minibatch_sequences
            ):
                selected = order[start : start + config.minibatch_sequences]
                loss, metrics = ppo_loss(
                    model,
                    {key: value[selected] for key, value in rollout.items()},
                    config,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                updates += 1
        record = {
            "kind": "training",
            "iteration": iteration,
            "agent_steps": iteration * steps_per_iteration,
            "gradient_updates": iteration * updates,
            "sampled_reward_mean": float(rollout["rewards"].mean().cpu()),
            **{
                key: float((value / updates).cpu())
                for key, value in totals.items()
            },
            "wall_seconds": time.monotonic() - started,
        }
        history.append(record)
        outputs.append_result(record)
        if (
            iteration % config.checkpoint_every_iterations == 0
            or iteration == iterations
        ):
            save(iteration)

    retained_checkpoints = [
        (iteration * steps_per_iteration, retained(iteration))
        for iteration in range(iterations + 1)
        if retained(iteration).exists()
    ]
    return history, retained_checkpoints
