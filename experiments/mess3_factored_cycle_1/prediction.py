"""Matched next-symbol twins trained on each RL agent's own trajectories."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from analysis.checkpoints import load_algorithm
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models.next_token import NextTokenAuxHead
from learners.models.transformer import TransformerModel


class PredictionTwin(NextTokenAuxHead, TransformerModel):
    """The PPO encoder architecture with an offline next-symbol head."""


def _device(context: RunContext) -> torch.device:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def _initial_state(module: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value).unsqueeze(0).to(device)
        for key, value in module.get_initial_state().items()
    }


@torch.no_grad()
def collect_policy_episodes(
    checkpoint: Path,
    env_factory: Callable[[], gym.Env],
    *,
    n_episodes: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Collect fixed-length episodes from the restored stochastic PPO policy."""

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    episodes: list[np.ndarray] = []
    action_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        module = module.to(device).eval()
        env = env_factory()
        try:
            for episode_index in range(n_episodes):
                observation, _ = env.reset(seed=seed + episode_index)
                state = _initial_state(module, device)
                recorded = [np.asarray(observation, dtype=np.float32)]
                done = False
                while not done:
                    observation_tensor = torch.from_numpy(
                        np.asarray(observation, dtype=np.float32)
                    ).unsqueeze(0).to(device)
                    embedding, state = module.encode_step(observation_tensor, state)
                    logits = module.action_distribution_inputs(embedding)
                    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                    action = int(
                        action_rng.choice(len(probabilities), p=probabilities)
                    )
                    observation, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    if not done:
                        recorded.append(
                            np.asarray(observation, dtype=np.float32)
                        )
                episodes.append(np.stack(recorded))
        finally:
            env.close()
    lengths = {len(episode) for episode in episodes}
    if len(lengths) != 1:
        raise ValueError("prediction-twin collection requires fixed episodes")
    return np.stack(episodes)


def _joint_token_targets(
    observations: np.ndarray,
    *,
    token_encoding: str,
) -> np.ndarray:
    if token_encoding == "joint":
        return observations[..., :9].argmax(axis=-1).astype(np.int64)
    if token_encoding == "factored":
        first = observations[..., :3].argmax(axis=-1)
        second = observations[..., 3:6].argmax(axis=-1)
        return (3 * first + second).astype(np.int64)
    raise ValueError(f"unknown token encoding {token_encoding!r}")


def _sequence_chunks(
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    lookback: int,
    chunk_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build padded BPTT chunks with exact preceding transformer context."""

    contexts, lengths, chunks, chunk_targets, masks = [], [], [], [], []
    width = observations.shape[-1]
    for episode, episode_targets in zip(observations, targets):
        for start in range(0, len(episode) - 1, chunk_length):
            end = min(start + chunk_length, len(episode) - 1)
            count = end - start
            history = episode[max(0, start - lookback) : start]
            context = np.zeros((lookback, width), dtype=np.float32)
            if len(history):
                context[-len(history) :] = history
            chunk = np.zeros((chunk_length, width), dtype=np.float32)
            chunk[:count] = episode[start:end]
            labels = np.zeros(chunk_length, dtype=np.int64)
            labels[:count] = episode_targets[start + 1 : end + 1]
            mask = np.zeros(chunk_length, dtype=bool)
            mask[:count] = True
            contexts.append(context)
            lengths.append(len(history))
            chunks.append(chunk)
            chunk_targets.append(labels)
            masks.append(mask)
    return (
        np.stack(contexts),
        np.asarray(lengths, dtype=np.float32),
        np.stack(chunks),
        np.stack(chunk_targets),
        np.stack(masks),
    )


def train_prediction_twin(
    context: RunContext,
    *,
    checkpoint: Path,
    env_factory: Callable[[], gym.Env],
    model_config: dict[str, Any],
    token_encoding: str,
    data_steps: int,
    epochs: int = 6,
    learning_rate: float = 3e-4,
) -> dict[str, Any]:
    """Train the matched predictor and write compact provenance/results."""

    if context.seed is None:
        raise ValueError("prediction twins require a resolved seed")
    if data_steps <= 1 or epochs <= 0:
        raise ValueError("prediction data_steps and epochs must be positive")
    device = _device(context)
    probe_env = env_factory()
    try:
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
        episode_length = int(probe_env.unwrapped.config.episode_length)
    finally:
        probe_env.close()
    n_episodes = max(1, math.ceil(data_steps / episode_length))
    observations = collect_policy_episodes(
        checkpoint,
        env_factory,
        n_episodes=n_episodes,
        seed=context.seed + 10_000,
        device=device,
    )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    dataset_path = context.artifacts_dir / "rl_policy_trajectories.npz"
    np.savez_compressed(dataset_path, observations=observations)

    torch.manual_seed(context.seed + 20_000)
    resolved_model_config = {
        **model_config,
        "next_token_aux": {"num_classes": 9},
    }
    module = PredictionTwin(
        observation_space=observation_space,
        action_space=action_space,
        model_config=resolved_model_config,
    ).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    targets = _joint_token_targets(
        observations,
        token_encoding=token_encoding,
    )
    chunk_length = int(resolved_model_config.get("max_seq_len", 32))
    contexts, lengths, chunks, labels, masks = _sequence_chunks(
        observations,
        targets,
        lookback=module.sequence_lookback,
        chunk_length=chunk_length,
    )
    checkpoints = context.artifacts_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    def save(name: str, optimizer_step: int) -> Path:
        path = checkpoints / f"prediction_twin_{name}.pt"
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in module.state_dict().items()
                },
                "optimizer_step": optimizer_step,
                "model_config": resolved_model_config,
                "token_encoding": token_encoding,
            },
            path,
        )
        return path

    save("init", 0)
    rng = np.random.default_rng(context.seed + 30_000)
    optimizer_step = 0
    final_loss = float("nan")
    final_accuracy = float("nan")
    batch_chunks = 8 if context.smoke else 64
    progress_path = context.artifacts_dir / "prediction_progress.jsonl"
    with progress_path.open("w") as progress:
        for epoch in range(epochs):
            order = rng.permutation(len(chunks))
            for start in range(0, len(order), batch_chunks):
                indices = order[start : start + batch_chunks]
                context_tensor = torch.from_numpy(contexts[indices]).to(device)
                length_tensor = torch.from_numpy(lengths[indices]).to(device)
                chunk_tensor = torch.from_numpy(chunks[indices]).to(device)
                label_tensor = torch.from_numpy(labels[indices]).to(device)
                mask_tensor = torch.from_numpy(masks[indices]).to(device)
                embeddings = module.encode_chunks(
                    context_tensor,
                    length_tensor,
                    chunk_tensor,
                )
                logits = module.next_token_aux_head(embeddings)
                loss = F.cross_entropy(logits[mask_tensor], label_tensor[mask_tensor])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_step += 1
                with torch.no_grad():
                    accuracy = (
                        logits[mask_tensor].argmax(dim=-1)
                        == label_tensor[mask_tensor]
                    ).float().mean()
                final_loss = float(loss.detach())
                final_accuracy = float(accuracy)
                progress.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "optimizer_step": optimizer_step,
                            "cross_entropy": final_loss,
                            "accuracy": final_accuracy,
                        }
                    )
                    + "\n"
                )
    final_checkpoint = save("final", optimizer_step)
    summary = {
        "objective": "next_joint_symbol_prediction",
        "trajectory_source": "restored_stochastic_rl_policy",
        "rl_checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "episodes": n_episodes,
        "observations": int(observations.shape[0] * observations.shape[1]),
        "epochs": epochs,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "optimizer_steps": optimizer_step,
        "final_cross_entropy": final_loss,
        "final_accuracy": final_accuracy,
        "final_checkpoint": str(final_checkpoint),
    }
    outputs.write_json("prediction_twin_summary.json", summary)
    return summary


def twin_context(context: RunContext) -> RunContext:
    """Place matched-predictor outputs below their owning RL run."""

    return replace(
        context,
        results_dir=context.results_dir / "prediction_twin",
        artifacts_dir=context.artifacts_dir / "prediction_twin",
        resume_from=None,
    )
