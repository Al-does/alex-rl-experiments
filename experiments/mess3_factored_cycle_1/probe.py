"""Held-out final-stream probes and E3 function-coupling diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from analysis.checkpoints import load_algorithm
from analysis.probes import (
    fit_affine_probe,
    global_mse_metrics,
    probe_predict,
    r2_score,
)
from analysis.rollouts import collect_batched_rollout_data
from experiments.mess3_factored_cycle_1.analysis import (
    geometry_report,
    nested_function_features,
)
from experiments.mess3_factored_cycle_1.dynamics import (
    action_kernels,
    reward_vector,
    value_iteration,
)
from experiments.mess3_factored_cycle_1.reference import factor_targets
from harness.context import RunContext
from harness.hardware import PROFILES

if TYPE_CHECKING:
    from experiments.mess3_factored_cycle_1.shared import Condition


@dataclass(frozen=True, slots=True)
class ProbeBudget:
    train_steps: int
    test_steps: int
    n_envs: int
    warmup: int

    @classmethod
    def for_context(cls, context: RunContext) -> "ProbeBudget":
        if context.smoke:
            return cls(train_steps=512, test_steps=512, n_envs=4, warmup=4)
        return cls(
            train_steps=60_000,
            test_steps=80_000,
            n_envs=16,
            warmup=64,
        )


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    beliefs: np.ndarray
    action_logits: np.ndarray
    values: np.ndarray


def _device(context: RunContext) -> torch.device:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _initial_state(
    module: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        .to(device)
        for key, value in module.get_initial_state().items()
    }


@torch.no_grad()
def _collect(
    module: Any,
    env_factory,
    *,
    n_steps: int,
    n_envs: int,
    warmup: int,
    seed: int,
    device: torch.device,
) -> ProbeData:
    module = module.to(device).eval()

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices: np.ndarray):
        fresh = _initial_state(module, len(indices), device)
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    def step_adapter(observations, state, randomness, action_spaces):
        del randomness, action_spaces
        observation = torch.from_numpy(observations).float().to(device)
        embedding, state = module.encode_step(observation, state)
        logits = module.action_distribution_inputs(embedding)
        values = module.heads.values(embedding).reshape(-1, 1)
        actions = logits.argmax(dim=-1).cpu().numpy()
        payload = torch.cat([embedding, logits, values], dim=-1).cpu().numpy()
        return actions, state, payload

    def target_adapter(observations, infos, episode_steps):
        del observations, episode_steps
        return {
            "beliefs": np.stack([info["belief_current"] for info in infos]),
        }

    collected = collect_batched_rollout_data(
        env_factory,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=n_envs,
        initial_state=initial_state,
        reset_state=reset_state,
        warmup=warmup,
    )
    action_count = int(module.action_space.n)
    representation = np.asarray(collected.representations, dtype=np.float64)
    embedding_width = representation.shape[1] - action_count - 1
    return ProbeData(
        activations=representation[:, :embedding_width],
        action_logits=representation[
            :, embedding_width : embedding_width + action_count
        ],
        values=representation[:, -1:],
        beliefs=np.asarray(collected.targets["beliefs"], dtype=np.float64),
    )


def _target_probe(
    train_activations: np.ndarray,
    train_target: np.ndarray,
    test_activations: np.ndarray,
    test_target: np.ndarray,
) -> dict[str, float]:
    weight, bias = fit_affine_probe(
        train_activations,
        train_target,
        ridge=1e-6,
    )
    predicted = probe_predict(weight, bias, test_activations)
    return {
        **global_mse_metrics(predicted, test_target),
        "r_squared": r2_score(predicted, test_target),
    }


def _nested_readout(
    train_features: dict[str, np.ndarray],
    test_features: dict[str, np.ndarray],
    train_target: np.ndarray,
    test_target: np.ndarray,
) -> dict[str, Any]:
    reports = {}
    for name in ("factor_only", "with_joint_interactions"):
        weight, bias = fit_affine_probe(
            train_features[name],
            train_target,
            ridge=1e-8,
        )
        predicted = probe_predict(weight, bias, test_features[name])
        reports[name] = {
            **global_mse_metrics(predicted, test_target),
            "r_squared": r2_score(predicted, test_target),
        }
    reports["interaction_delta_r_squared"] = float(
        reports["with_joint_interactions"]["r_squared"]
        - reports["factor_only"]["r_squared"]
    )
    return reports


def probe_checkpoint(
    context: RunContext,
    checkpoint: Path,
    condition: "Condition",
    *,
    budget: ProbeBudget | None = None,
) -> dict[str, Any]:
    """Probe one RL checkpoint on independent train/test policy rollouts."""

    if context.seed is None:
        raise ValueError("factored probes require a resolved seed")
    budget = budget or ProbeBudget.for_context(context)
    device = _device(context)
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        env_config["diagnostics"] = {"belief": True}

        def env_factory():
            return env_class(env_config)

        train = _collect(
            module,
            env_factory,
            n_steps=budget.train_steps,
            n_envs=budget.n_envs,
            warmup=budget.warmup,
            seed=context.seed + 40_000,
            device=device,
        )
        test = _collect(
            module,
            env_factory,
            n_steps=budget.test_steps,
            n_envs=budget.n_envs,
            warmup=budget.warmup,
            seed=context.seed + 50_000,
            device=device,
        )

    train_targets = factor_targets(train.beliefs)
    test_targets = factor_targets(test.beliefs)
    target_names = [
        "joint",
        "f1",
        "f2",
        "f2_goal_block",
        "f2_within_n",
        "relative_phase",
    ]
    probes = {
        name: _target_probe(
            train.activations,
            train_targets[name],
            test.activations,
            test_targets[name],
        )
        for name in target_names
    }
    conditional = (
        test_targets["f2_non_goal_mass"][:, 0] >= 0.2
    )
    if int(conditional.sum()) >= 10:
        weight, bias = fit_affine_probe(
            train.activations[
                train_targets["f2_non_goal_mass"][:, 0] >= 0.2
            ],
            train_targets["f2_within_n_conditional"][
                train_targets["f2_non_goal_mass"][:, 0] >= 0.2
            ],
            ridge=1e-6,
        )
        predicted = probe_predict(weight, bias, test.activations[conditional])
        probes["f2_within_n_conditional"] = {
            **global_mse_metrics(
                predicted,
                test_targets["f2_within_n_conditional"][conditional],
            ),
            "r_squared": r2_score(
                predicted,
                test_targets["f2_within_n_conditional"][conditional],
            ),
            "n_test": int(conditional.sum()),
        }

    function_coupling = None
    if condition.experiment in {"E3a", "E3b", "E3c"}:
        train_features = nested_function_features(
            train_targets["f1"],
            train_targets["f2"],
        )
        test_features = nested_function_features(
            test_targets["f1"],
            test_targets["f2"],
        )
        solved = value_iteration(
            action_kernels(
                condition.action_kind,
                coupling_lambda=condition.coupling_lambda,
            ),
            reward_vector(condition.reward_kind),
        )
        train_q = train.beliefs @ solved.q_values
        test_q = test.beliefs @ solved.q_values
        function_coupling = {
            "exact_qmdp_value": _nested_readout(
                train_features,
                test_features,
                train_q.max(axis=1, keepdims=True),
                test_q.max(axis=1, keepdims=True),
            ),
            "exact_qmdp_centered_action_scores": _nested_readout(
                train_features,
                test_features,
                train_q - train_q.mean(axis=1, keepdims=True),
                test_q - test_q.mean(axis=1, keepdims=True),
            ),
            "model_value": _nested_readout(
                train_features,
                test_features,
                train.values,
                test.values,
            ),
            "model_centered_action_logits": _nested_readout(
                train_features,
                test_features,
                train.action_logits
                - train.action_logits.mean(axis=1, keepdims=True),
                test.action_logits
                - test.action_logits.mean(axis=1, keepdims=True),
            ),
        }

    return {
        "condition": asdict(condition),
        "checkpoint": str(checkpoint),
        "budget": asdict(budget),
        "representation": "post_final_layer_norm",
        "probes": probes,
        "geometry": geometry_report(
            train.activations,
            train.beliefs,
            expected_quotient_dimension=condition.expected_quotient_dimension,
        ),
        "function_coupling": function_coupling,
        "interpretation": (
            "Held-out decodability supports geometry interpretation but does "
            "not by itself establish causal policy use."
        ),
    }
