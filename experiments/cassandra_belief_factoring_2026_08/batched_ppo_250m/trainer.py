"""Single-process, device-batched PPO for the Cassandra throughput study."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from harness.artifacts import RunArtifacts
from harness.context import RunContext

from .environment import ActionScope, BatchedCassandraEnv, OBSERVATION_DIM
from .model import BatchedTransformerActorCritic, InferenceState


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    action_scope: ActionScope
    total_env_steps: int
    num_envs: int
    rollout_steps: int
    minibatch_size: int
    num_epochs: int
    episode_length: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_param: float
    vf_clip_param: float
    vf_loss_coeff: float
    entropy_coeff: float
    d_model: int
    n_layers: int
    n_heads: int
    context_len: int
    checkpoint_interval: int
    compile_model: bool = False
    bucket_sequences: bool = False
    cache_inference_constants: bool = False
    reuse_environment_buffers: bool = False
    use_bfloat16: bool = False

    def __post_init__(self) -> None:
        if self.total_env_steps <= 0:
            raise ValueError("total_env_steps must be positive")
        if self.num_envs <= 0 or self.rollout_steps <= 0:
            raise ValueError("num_envs and rollout_steps must be positive")
        if self.minibatch_size < self.rollout_steps:
            raise ValueError(
                "minibatch_size must fit at least one complete sequence"
            )

    @property
    def train_batch_size(self) -> int:
        return self.num_envs * self.rollout_steps


@dataclass
class Rollout:
    start_context: torch.Tensor
    start_context_length: int
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    boundaries: torch.Tensor
    sampling_seconds: float
    completed_return_mean: float | None


@dataclass
class TrainingBatch:
    contexts: torch.Tensor
    context_lengths: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    loss_mask: torch.Tensor


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    final_values: torch.Tensor,
    boundaries: torch.Tensor,
    truncated_bootstraps: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Compute GAE while bootstrapping, but not crossing, time-limit resets."""

    horizon, num_envs = rewards.shape
    advantages = torch.empty_like(rewards)
    gae = torch.zeros(num_envs, device=rewards.device)
    for timestep in range(horizon - 1, -1, -1):
        next_values = (
            final_values if timestep == horizon - 1 else values[timestep + 1]
        )
        next_values = torch.where(
            boundaries[timestep],
            truncated_bootstraps[timestep],
            next_values,
        )
        delta = rewards[timestep] + gamma * next_values - values[timestep]
        continuation = (~boundaries[timestep]).to(rewards.dtype)
        gae = (
            delta
            + gamma * gae_lambda * continuation * gae
        )
        advantages[timestep] = gae
    return advantages


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class BatchedPPOTrainer:
    """Own rollout, optimization, checkpoint, and compact metric lifecycles."""

    def __init__(
        self,
        *,
        config: TrainerConfig,
        context: RunContext,
        device: torch.device,
    ) -> None:
        if context.seed is None:
            raise ValueError("batched Cassandra PPO requires a resolved seed")
        self.config = config
        self.context = context
        self.device = device
        self.outputs = RunArtifacts.from_context(context)
        self.outputs.prepare()

        torch.manual_seed(context.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(context.seed)
        self.environment = BatchedCassandraEnv(
            num_envs=config.num_envs,
            action_scope=config.action_scope,
            episode_length=config.episode_length,
            seed=context.seed,
            device=device,
            reuse_buffers=config.reuse_environment_buffers,
        )
        self.model = BatchedTransformerActorCritic(
            observation_dim=OBSERVATION_DIM,
            action_count=self.environment.action_count,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            context_len=config.context_len,
            cache_inference_constants=config.cache_inference_constants,
        ).to(device)
        if config.compile_model:
            if device.type != "cuda":
                raise ValueError("model compilation is enabled only for CUDA")
            self.model.enable_compilation()
        self.model_dtype = (
            torch.bfloat16 if config.use_bfloat16 else torch.float32
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.action_generator = torch.Generator(device=device)
        self.action_generator.manual_seed(context.seed + 2_000_003)
        self.shuffle_generator = torch.Generator(device=device)
        self.shuffle_generator.manual_seed(context.seed + 3_000_003)
        self.observations = self.environment.reset()
        self.inference_state = self.model.initial_state(
            config.num_envs, device, dtype=self.model_dtype
        )
        self.total_env_steps = 0
        self.iteration = 0
        self.next_checkpoint = config.checkpoint_interval
        self.latest_episode_return: float | None = None

        if context.resume_from is not None:
            self.load_checkpoint(context.resume_from)

    def _sample_actions(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probabilities = logits.log_softmax(dim=-1)
        actions = torch.multinomial(
            log_probabilities.exp(),
            num_samples=1,
            generator=self.action_generator,
        ).squeeze(-1)
        selected = log_probabilities.gather(
            dim=-1, index=actions[:, None]
        ).squeeze(-1)
        return actions, selected

    def collect_rollout(self) -> Rollout:
        config = self.config
        n, horizon = config.num_envs, config.rollout_steps
        self.model.eval()
        start_context = self.model.ordered_context(self.inference_state)
        start_context_length = self.inference_state.context_length
        observations = torch.empty(
            (horizon, n, OBSERVATION_DIM),
            device=self.device,
            dtype=self.model_dtype,
        )
        actions = torch.empty(
            (horizon, n), dtype=torch.long, device=self.device
        )
        old_log_probs = torch.empty((horizon, n), device=self.device)
        old_values = torch.empty((horizon, n), device=self.device)
        rewards = torch.empty((horizon, n), device=self.device)
        boundaries = torch.zeros(
            horizon, dtype=torch.bool, device=self.device
        )
        truncated_bootstraps = torch.zeros(
            (horizon, n), device=self.device
        )
        completed_returns: list[float] = []

        _sync(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            for timestep in range(horizon):
                model_observations = self.observations.to(self.model_dtype)
                observations[timestep].copy_(model_observations)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.config.use_bfloat16,
                ):
                    logits, values, self.inference_state = (
                        self.model.inference(
                            model_observations, self.inference_state
                        )
                    )
                logits = logits.float()
                values = values.float()
                sampled_actions, log_probs = self._sample_actions(logits)
                next_observations, step_rewards, truncated = (
                    self.environment.step(sampled_actions)
                )
                actions[timestep].copy_(sampled_actions)
                old_log_probs[timestep].copy_(log_probs)
                old_values[timestep].copy_(values)
                rewards[timestep].copy_(step_rewards)

                if truncated:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.config.use_bfloat16,
                    ):
                        _, bootstrap_values, _ = self.model.inference(
                            next_observations.to(self.model_dtype),
                            self.inference_state,
                            record_context=False,
                        )
                    truncated_bootstraps[timestep].copy_(bootstrap_values)
                    boundaries[timestep] = True
                    completed_returns.append(
                        float(self.environment.episode_returns.mean().item())
                    )
                    next_observations = self.environment.reset()
                    self.inference_state = self.model.initial_state(
                        n, self.device, dtype=self.model_dtype
                    )
                self.observations = next_observations

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.config.use_bfloat16,
            ):
                _, final_values, _ = self.model.inference(
                    self.observations.to(self.model_dtype),
                    self.inference_state,
                    record_context=False,
                )
            final_values = final_values.float()
        _sync(self.device)
        sampling_seconds = time.perf_counter() - started

        advantages = compute_gae(
            rewards=rewards,
            values=old_values,
            final_values=final_values,
            boundaries=boundaries,
            truncated_bootstraps=truncated_bootstraps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        return Rollout(
            start_context=start_context,
            start_context_length=start_context_length,
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            rewards=rewards,
            advantages=advantages,
            returns=advantages + old_values,
            boundaries=boundaries,
            sampling_seconds=sampling_seconds,
            completed_return_mean=(
                sum(completed_returns) / len(completed_returns)
                if completed_returns
                else None
            ),
        )

    def _padded_segment(
        self, values: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        horizon = self.config.rollout_steps
        segment = values[start:stop].transpose(0, 1)
        output = values.new_zeros(
            (self.config.num_envs, horizon, *values.shape[2:])
        )
        output[:, : stop - start].copy_(segment)
        return output

    def prepare_training_batch(self, rollout: Rollout) -> TrainingBatch:
        boundary_steps = (
            rollout.boundaries.nonzero().flatten().add(1).tolist()
        )
        ends = [*boundary_steps, self.config.rollout_steps]
        starts = [0, *boundary_steps]
        segments = [
            (start, stop)
            for start, stop in zip(starts, ends)
            if stop > start
        ]
        contexts: list[torch.Tensor] = []
        context_lengths: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        columns: dict[str, list[torch.Tensor]] = {
            "observations": [],
            "actions": [],
            "old_log_probs": [],
            "old_values": [],
            "advantages": [],
            "returns": [],
        }
        for index, (start, stop) in enumerate(segments):
            if index == 0:
                contexts.append(rollout.start_context)
                context_length = rollout.start_context_length
            else:
                contexts.append(torch.zeros_like(rollout.start_context))
                context_length = 0
            context_lengths.append(
                torch.full(
                    (self.config.num_envs,),
                    float(context_length),
                    device=self.device,
                )
            )
            mask = torch.zeros(
                (
                    self.config.num_envs,
                    self.config.rollout_steps,
                ),
                dtype=torch.bool,
                device=self.device,
            )
            mask[:, : stop - start] = True
            masks.append(mask)
            for name in columns:
                columns[name].append(
                    self._padded_segment(
                        getattr(rollout, name), start, stop
                    )
                )

        advantages = torch.cat(columns["advantages"])
        loss_mask = torch.cat(masks)
        valid_advantages = advantages[loss_mask]
        advantages = advantages.clone()
        advantages[loss_mask] = (
            valid_advantages - valid_advantages.mean()
        ) / (valid_advantages.std(unbiased=False) + 1e-8)
        return TrainingBatch(
            contexts=torch.cat(contexts),
            context_lengths=torch.cat(context_lengths),
            observations=torch.cat(columns["observations"]),
            actions=torch.cat(columns["actions"]),
            old_log_probs=torch.cat(columns["old_log_probs"]),
            old_values=torch.cat(columns["old_values"]),
            advantages=advantages,
            returns=torch.cat(columns["returns"]),
            loss_mask=loss_mask,
        )

    def prepare_training_batches(
        self, rollout: Rollout
    ) -> list[TrainingBatch]:
        if not self.config.bucket_sequences:
            return [self.prepare_training_batch(rollout)]

        normalized_advantages = rollout.advantages.clone()
        flat_advantages = normalized_advantages.flatten()
        normalized_advantages = (
            normalized_advantages - flat_advantages.mean()
        ) / (flat_advantages.std(unbiased=False) + 1e-8)
        boundary_steps = (
            rollout.boundaries.nonzero().flatten().add(1).tolist()
        )
        ends = [*boundary_steps, self.config.rollout_steps]
        starts = [0, *boundary_steps]
        batches: list[TrainingBatch] = []
        for index, (start, stop) in enumerate(zip(starts, ends)):
            if stop <= start:
                continue
            length = stop - start
            context = (
                rollout.start_context
                if index == 0
                else torch.zeros_like(rollout.start_context)
            )
            context_length = (
                rollout.start_context_length if index == 0 else 0
            )
            batches.append(
                TrainingBatch(
                    contexts=context,
                    context_lengths=torch.full(
                        (self.config.num_envs,),
                        float(context_length),
                        device=self.device,
                    ),
                    observations=rollout.observations[
                        start:stop
                    ].transpose(0, 1),
                    actions=rollout.actions[start:stop].transpose(0, 1),
                    old_log_probs=rollout.old_log_probs[
                        start:stop
                    ].transpose(0, 1),
                    old_values=rollout.old_values[
                        start:stop
                    ].transpose(0, 1),
                    advantages=normalized_advantages[
                        start:stop
                    ].transpose(0, 1),
                    returns=rollout.returns[start:stop].transpose(0, 1),
                    loss_mask=torch.ones(
                        (self.config.num_envs, length),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                )
            )
        return batches

    def optimize(
        self, batches: list[TrainingBatch]
    ) -> dict[str, float]:
        config = self.config
        totals = {
            "loss": torch.zeros((), device=self.device),
            "policy_loss": torch.zeros((), device=self.device),
            "value_loss": torch.zeros((), device=self.device),
            "entropy": torch.zeros((), device=self.device),
            "approx_kl": torch.zeros((), device=self.device),
            "clip_fraction": torch.zeros((), device=self.device),
            "valid": torch.zeros((), device=self.device),
        }
        self.model.train()
        _sync(self.device)
        started = time.perf_counter()
        for _ in range(config.num_epochs):
            for batch in batches:
                sequence_length = batch.observations.shape[1]
                sequences_per_minibatch = max(
                    1, config.minibatch_size // sequence_length
                )
                permutation = torch.randperm(
                    batch.contexts.shape[0],
                    generator=self.shuffle_generator,
                    device=self.device,
                )
                for offset in range(
                    0, permutation.numel(), sequences_per_minibatch
                ):
                    indices = permutation[
                        offset : offset + sequences_per_minibatch
                    ]
                    mask = batch.loss_mask[indices]
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.config.use_bfloat16,
                    ):
                        logits, values = self.model.training_outputs(
                            batch.contexts[indices],
                            batch.context_lengths[indices],
                            batch.observations[indices],
                        )
                    logits = logits.float()[mask]
                    values = values.float()[mask]
                    minibatch_actions = batch.actions[indices][mask]
                    old_log_probs = batch.old_log_probs[indices][mask]
                    advantages = batch.advantages[indices][mask]
                    returns = batch.returns[indices][mask]

                    log_probs_all = logits.log_softmax(dim=-1)
                    probabilities = log_probs_all.exp()
                    log_probs = log_probs_all.gather(
                        -1, minibatch_actions[:, None]
                    ).squeeze(-1)
                    entropy = -(probabilities * log_probs_all).sum(dim=-1)
                    log_ratio = log_probs - old_log_probs
                    ratio = log_ratio.exp()
                    surrogate = torch.minimum(
                        advantages * ratio,
                        advantages
                        * ratio.clamp(
                            1.0 - config.clip_param,
                            1.0 + config.clip_param,
                        ),
                    )
                    squared_value_error = (values - returns).square()
                    clipped_value_error = squared_value_error.clamp(
                        max=config.vf_clip_param
                    )
                    loss = (
                        -surrogate
                        + config.vf_loss_coeff * clipped_value_error
                        - config.entropy_coeff * entropy
                    ).mean()

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                    count = torch.as_tensor(
                        logits.shape[0],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    totals["loss"] += loss.detach() * count
                    totals["policy_loss"] += -surrogate.detach().sum()
                    totals["value_loss"] += (
                        clipped_value_error.detach().sum()
                    )
                    totals["entropy"] += entropy.detach().sum()
                    totals["approx_kl"] += (
                        (ratio - 1.0) - log_ratio
                    ).detach().sum()
                    totals["clip_fraction"] += (
                        (ratio - 1.0).abs() > config.clip_param
                    ).to(torch.float32).sum()
                    totals["valid"] += count
        _sync(self.device)
        learning_seconds = time.perf_counter() - started
        denominator = totals.pop("valid").clamp_min(1.0)
        metrics = {
            name: float((value / denominator).item())
            for name, value in totals.items()
        }
        metrics["learning_seconds"] = learning_seconds
        return metrics

    def _checkpoint_payload(self) -> dict[str, Any]:
        state = self.inference_state
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment": self.environment.state_dict(),
            "observations": self.observations,
            "inference_state": {
                "kv_k": state.kv_k,
                "kv_v": state.kv_v,
                "kv_len": state.kv_len,
                "raw_context": state.raw_context,
                "context_position": state.context_position,
                "context_length": state.context_length,
            },
            "action_rng": self.action_generator.get_state(),
            "shuffle_rng": self.shuffle_generator.get_state(),
            "total_env_steps": self.total_env_steps,
            "iteration": self.iteration,
            "next_checkpoint": self.next_checkpoint,
            "latest_episode_return": self.latest_episode_return,
        }

    def save_checkpoint(self, *, label: str) -> Path:
        directory = self.outputs.checkpoints_dir / label
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "checkpoint.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(self._checkpoint_payload(), temporary)
        temporary.replace(destination)
        index_path = self.outputs.checkpoints_dir / "index.json"
        records: list[dict[str, Any]] = []
        if index_path.is_file():
            records = json.loads(index_path.read_text()).get(
                "checkpoints", []
            )
        records = [
            record
            for record in records
            if record.get("label") != label
        ]
        records.append(
            {
                "label": label,
                "path": str(destination),
                "iteration": self.iteration,
                "env_steps": self.total_env_steps,
            }
        )
        temporary_index = index_path.with_suffix(".json.tmp")
        temporary_index.write_text(
            json.dumps({"checkpoints": records}, indent=2, sort_keys=True)
            + "\n"
        )
        temporary_index.replace(index_path)
        return destination

    def load_checkpoint(self, checkpoint: Path) -> None:
        path = Path(checkpoint)
        if path.is_dir():
            path = path / "checkpoint.pt"
        payload = torch.load(
            path, map_location=self.device, weights_only=True
        )
        if payload["config"] != asdict(self.config):
            raise ValueError("checkpoint trainer config does not match recipe")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.environment.load_state_dict(payload["environment"])
        self.observations = payload["observations"].to(self.device)
        state = payload["inference_state"]
        self.inference_state = InferenceState(
            kv_k=state["kv_k"].to(self.device),
            kv_v=state["kv_v"].to(self.device),
            kv_len=state["kv_len"].to(self.device),
            raw_context=state["raw_context"].to(self.device),
            context_position=int(state["context_position"]),
            context_length=int(state["context_length"]),
        )
        self.action_generator.set_state(payload["action_rng"].cpu())
        self.shuffle_generator.set_state(payload["shuffle_rng"].cpu())
        self.total_env_steps = int(payload["total_env_steps"])
        self.iteration = int(payload["iteration"])
        self.next_checkpoint = int(payload["next_checkpoint"])
        self.latest_episode_return = payload["latest_episode_return"]

    def train(self) -> dict[str, Any]:
        run_started = time.perf_counter()
        latest_checkpoint: Path | None = None
        while self.total_env_steps < self.config.total_env_steps:
            rollout = self.collect_rollout()
            training = self.optimize(
                self.prepare_training_batches(rollout)
            )
            self.iteration += 1
            self.total_env_steps += self.config.train_batch_size
            if rollout.completed_return_mean is not None:
                self.latest_episode_return = (
                    rollout.completed_return_mean
                )
            update_seconds = (
                rollout.sampling_seconds + training["learning_seconds"]
            )
            metrics = {
                "iteration": self.iteration,
                "env_steps": self.total_env_steps,
                "episode_return_mean": self.latest_episode_return,
                "sampling_seconds": rollout.sampling_seconds,
                "learning_seconds": training["learning_seconds"],
                "steps_per_second": (
                    self.config.train_batch_size / update_seconds
                ),
                "sampling_steps_per_second": (
                    self.config.train_batch_size
                    / rollout.sampling_seconds
                ),
                **{
                    name: value
                    for name, value in training.items()
                    if name != "learning_seconds"
                },
            }
            self.outputs.append_result(metrics)
            self.outputs.append_jsonl(
                "training_curves.jsonl", metrics, dest="results"
            )

            if self.total_env_steps >= self.next_checkpoint:
                while self.next_checkpoint <= self.total_env_steps:
                    self.next_checkpoint += self.config.checkpoint_interval
                latest_checkpoint = self.save_checkpoint(
                    label=f"steps_{self.total_env_steps:012d}"
                )

        latest_checkpoint = self.save_checkpoint(label="final")
        _sync(self.device)
        elapsed = time.perf_counter() - run_started
        return {
            "status": "completed",
            "iterations": self.iteration,
            "env_steps": self.total_env_steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": self.total_env_steps / elapsed,
            "episode_return_mean": self.latest_episode_return,
            "checkpoint": str(latest_checkpoint),
        }


__all__ = [
    "BatchedPPOTrainer",
    "Rollout",
    "TrainerConfig",
    "TrainingBatch",
    "compute_gae",
]
