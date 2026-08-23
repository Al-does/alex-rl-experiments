"""Measure whether the final targeted policy aims or shotguns maintenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

from envs.cassandra_machine import (
    Condition,
    N_COMPONENTS,
    TargetedAction,
    action_names,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext


FULL_STEPS = 64_000
SMOKE_STEPS = 4_096
N_ENVS = 8
CHECKPOINT = (
    Path("targeted_ppo_small_continue_30m")
    / "artifacts"
    / "20260820T205945Z-e2f72153"
    / "checkpoints"
    / "iteration_000916_final"
)


def checkpoint_path(context: RunContext) -> Path:
    return context.experiment_dir.parent / CHECKPOINT


def _initial_state(module: Any, batch: int, device: torch.device):
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch, *([1] * value.ndim))
        .to(device)
        for key, value in module.get_initial_state().items()
    }


def summarize_streaks(streaks: list[list[tuple[str, int]]]) -> dict[str, Any]:
    """Summarize contiguous maintenance streaks and four-component sweeps."""

    if not streaks:
        return {
            "count": 0,
            "mean_length": 0.0,
            "max_length": 0,
            "length_at_least_four_fraction": 0.0,
            "all_four_components_fraction": 0.0,
            "four_action_sweep_fraction": 0.0,
            "same_kind_all_four_fraction": 0.0,
        }
    lengths = np.asarray([len(streak) for streak in streaks])
    all_four = []
    any_window = []
    same_kind = []
    for streak in streaks:
        components = {component for _, component in streak}
        all_four.append(len(components) == N_COMPONENTS)
        any_window.append(
            any(
                len({component for _, component in streak[start : start + 4]})
                == N_COMPONENTS
                for start in range(max(len(streak) - 3, 0))
            )
        )
        same_kind.append(
            any(
                {
                    component
                    for action_kind, component in streak
                    if action_kind == kind
                }
                == set(range(N_COMPONENTS))
                for kind in ("repair", "replace")
            )
        )
    return {
        "count": len(streaks),
        "mean_length": float(lengths.mean()),
        "max_length": int(lengths.max()),
        "length_at_least_four_fraction": float(np.mean(lengths >= 4)),
        "all_four_components_fraction": float(np.mean(all_four)),
        "four_action_sweep_fraction": float(np.mean(any_window)),
        "same_kind_all_four_fraction": float(np.mean(same_kind)),
    }


def run(context: RunContext):
    """Collect stochastic policy rollouts and target-selection diagnostics."""

    checkpoint = checkpoint_path(context)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"targeted checkpoint not found: {checkpoint}")
    n_steps = SMOKE_STEPS if context.smoke else FULL_STEPS
    rng = np.random.default_rng(context.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module_checkpoint = (
        checkpoint
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
    )
    module = RLModule.from_checkpoint(str(module_checkpoint)).to(device).eval()
    env_class = CassandraActionObservationEnv
    env_config = {
        "action_scope": "targeted",
        "initial_state_distribution": "all_good",
        "episode_length": 1_000,
        "diagnostics": True,
    }
    with torch.inference_mode():
        envs = [env_class(env_config) for _ in range(N_ENVS)]
        observations, infos = [], []
        episode_indices = np.zeros(N_ENVS, dtype=np.int64)
        episode_returns = np.zeros(N_ENVS, dtype=np.float64)
        completed_returns = []
        for index, env in enumerate(envs):
            observation, info = env.reset(seed=int(context.seed) + index)
            observations.append(observation)
            infos.append(info)
        state = _initial_state(module, N_ENVS, device)
        action_counts = np.zeros(len(TargetedAction), dtype=np.int64)
        greedy_counts = np.zeros(len(TargetedAction), dtype=np.int64)
        current_streaks: list[list[tuple[str, int]]] = [
            [] for _ in range(N_ENVS)
        ]
        completed_streaks: list[list[tuple[str, int]]] = []
        repair_records = []
        replace_records = []

        try:
            collected = 0
            while collected < n_steps:
                obs_tensor = torch.as_tensor(
                    np.stack(observations),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.inference_mode():
                    embeddings, state = module.encode_step(obs_tensor, state)
                    logits = module.action_distribution_inputs(embeddings)
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                actions = np.asarray(
                    [
                        rng.choice(len(row), p=row)
                        for row in probabilities
                    ],
                    dtype=np.int64,
                )
                greedy = probabilities.argmax(axis=1)

                for index, env in enumerate(envs):
                    action = int(actions[index])
                    action_counts[action] += 1
                    greedy_counts[int(greedy[index])] += 1
                    components = np.asarray(
                        infos[index]["components_current"],
                        dtype=np.int64,
                    )
                    marginals = np.asarray(
                        infos[index]["factored_belief_current"],
                        dtype=np.float64,
                    )
                    if int(TargetedAction.REPAIR_COMPONENT_0) <= action <= int(
                        TargetedAction.REPAIR_COMPONENT_3
                    ):
                        component = action - int(
                            TargetedAction.REPAIR_COMPONENT_0
                        )
                        useful = marginals[:, Condition.BAD].copy()
                        useful += marginals[:, Condition.FAIR]
                        repair_records.append(
                            {
                                "true_hit": components[component]
                                in (Condition.BAD, Condition.FAIR),
                                "chosen_belief": useful[component],
                                "mean_belief": useful.mean(),
                                "max_belief": useful.max(),
                            }
                        )
                        current_streaks[index].append(("repair", component))
                    elif int(
                        TargetedAction.REPLACE_COMPONENT_0
                    ) <= action <= int(TargetedAction.REPLACE_COMPONENT_3):
                        component = action - int(
                            TargetedAction.REPLACE_COMPONENT_0
                        )
                        broken = marginals[:, Condition.BROKEN]
                        condition_gain = (
                            marginals
                            @ (
                                int(Condition.GOOD)
                                - np.arange(len(Condition))
                            )
                        )
                        replace_records.append(
                            {
                                "true_broken_hit": components[component]
                                == Condition.BROKEN,
                                "true_non_good_hit": components[component]
                                != Condition.GOOD,
                                "chosen_broken_belief": broken[component],
                                "mean_broken_belief": broken.mean(),
                                "max_broken_belief": broken.max(),
                                "chosen_condition_gain": condition_gain[
                                    component
                                ],
                                "mean_condition_gain": condition_gain.mean(),
                                "max_condition_gain": condition_gain.max(),
                            }
                        )
                        current_streaks[index].append(("replace", component))
                    elif current_streaks[index]:
                        completed_streaks.append(current_streaks[index])
                        current_streaks[index] = []

                    next_obs, reward, terminated, truncated, next_info = env.step(
                        action
                    )
                    episode_returns[index] += reward
                    collected += 1
                    if terminated or truncated:
                        if current_streaks[index]:
                            completed_streaks.append(current_streaks[index])
                            current_streaks[index] = []
                        completed_returns.append(episode_returns[index])
                        episode_returns[index] = 0.0
                        episode_indices[index] += 1
                        next_obs, next_info = env.reset(
                            seed=(
                                int(context.seed)
                                + index
                                + int(episode_indices[index]) * N_ENVS
                            )
                        )
                        fresh = _initial_state(module, 1, device)
                        for key, value in state.items():
                            value[index : index + 1].copy_(fresh[key])
                    observations[index] = next_obs
                    infos[index] = next_info
                    if collected >= n_steps:
                        break
        finally:
            for env in envs:
                env.close()

    for streak in current_streaks:
        if streak:
            completed_streaks.append(streak)

    def aggregate(records, fields):
        return {
            "count": len(records),
            **{
                field: (
                    float(np.mean([record[field] for record in records]))
                    if records
                    else None
                )
                for field in fields
            },
        }

    result = {
        "checkpoint": checkpoint,
        "seed": context.seed,
        "steps": n_steps,
        "episodes_completed": len(completed_returns),
        "episode_return_mean": (
            float(np.mean(completed_returns)) if completed_returns else None
        ),
        "action_counts": action_counts,
        "action_fractions": action_counts / max(action_counts.sum(), 1),
        "action_fractions_by_name": {
            name: fraction
            for name, fraction in zip(
                action_names("targeted"),
                action_counts / max(action_counts.sum(), 1),
            )
        },
        "greedy_action_counts": greedy_counts,
        "greedy_action_fractions": greedy_counts / max(greedy_counts.sum(), 1),
        "greedy_action_fractions_by_name": {
            name: fraction
            for name, fraction in zip(
                action_names("targeted"),
                greedy_counts / max(greedy_counts.sum(), 1),
            )
        },
        "repair_targeting": aggregate(
            repair_records,
            ("true_hit", "chosen_belief", "mean_belief", "max_belief"),
        ),
        "replace_targeting": aggregate(
            replace_records,
            (
                "true_broken_hit",
                "true_non_good_hit",
                "chosen_broken_belief",
                "mean_broken_belief",
                "max_broken_belief",
                "chosen_condition_gain",
                "mean_condition_gain",
                "max_condition_gain",
            ),
        ),
        "maintenance_streaks": summarize_streaks(completed_streaks),
    }
    RunArtifacts.from_context(context).write_json(
        "targeted_policy_action_diagnostic.json",
        result,
    )
    return result


__all__ = ["CHECKPOINT", "checkpoint_path", "run", "summarize_streaks"]
