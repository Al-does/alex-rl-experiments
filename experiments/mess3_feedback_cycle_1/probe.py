"""Multi-target transducer probe data for guess-driven MESS3 feedback.

One rollout yields one activation matrix and several aligned Bayesian targets
that differ only in how much of the agent's own influence they account for:

``executed``
    The exact action-conditioned predictive belief. This is the transducer
    target of ``rl-harness``: ``K(x | a) = diag(P(x | s)) U(a)`` at delay one.
``blind``
    The same filter run as if every guess executed the passive kernel. A model
    that ignores its own feedback can do no better than this.
``marginal``
    The filter of the stacked, renormalized single HMM ``Ubar``, which knows
    the policy's guess statistics but not the realized guess.
``joint``
    The nine-state filter over the factored ``(m, Phi)`` lift, whose marginals
    are the two factor predictive vectors of the composition hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from analysis.probes import predictive_belief_update
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from experiments.mess3_feedback_cycle_1.dynamics import (
    executed_state_belief,
    factor_marginals,
    joint_emission,
    joint_initial_distribution,
    joint_transitions,
    product_state,
)
from harness.seeding import seed_sequence_to_int


TARGET_NAMES = ("executed", "blind", "marginal", "joint", "factor_m", "factor_phi")


@dataclass(frozen=True, slots=True)
class FeedbackProbeData:
    """Aligned activations and every Bayesian target of the feedback loop."""

    activations: np.ndarray
    executed: np.ndarray
    diagnostic: np.ndarray
    blind: np.ndarray
    marginal: np.ndarray | None
    joint: np.ndarray
    factor_m: np.ndarray
    factor_phi: np.ndarray
    tokens: np.ndarray
    previous_tokens: np.ndarray
    actions: np.ndarray
    previous_actions: np.ndarray
    states: np.ndarray
    env_indices: np.ndarray
    episode_steps: np.ndarray
    rewards: np.ndarray

    def target(self, name: str) -> np.ndarray | None:
        return getattr(self, name)

    def product_state_gap(self) -> dict[str, float]:
        """Report how far the joint belief sits from the product manifold."""

        factored = product_state(self.factor_m, self.factor_phi)
        return {
            "joint_product_mse": float(np.square(self.joint - factored).mean()),
            "executed_product_mse": float(
                np.square(
                    self.executed - executed_state_belief(factored)
                ).mean()
            ),
        }


@dataclass(frozen=True, slots=True)
class FeedbackFilters:
    """Every operator family needed to rebuild the targets during a rollout."""

    emission: np.ndarray
    initial_belief: np.ndarray
    blind_transition: np.ndarray
    joint_emission: np.ndarray
    joint_initial: np.ndarray
    joint_transitions: np.ndarray
    marginal_transition: np.ndarray | None = None

    def with_marginal(self, transition: np.ndarray | None) -> FeedbackFilters:
        return FeedbackFilters(
            emission=self.emission,
            initial_belief=self.initial_belief,
            blind_transition=self.blind_transition,
            joint_emission=self.joint_emission,
            joint_initial=self.joint_initial,
            joint_transitions=self.joint_transitions,
            marginal_transition=(
                None if transition is None else np.asarray(transition, dtype=np.float64)
            ),
        )


def make_feedback_filters(
    environment: Any,
    *,
    feedback_strength: float,
) -> FeedbackFilters:
    """Build every filter family from public environment data."""

    model = environment.model
    if environment.config.observation.token_scrambling != "none":
        raise ValueError("feedback probing requires unscrambled tokens")
    if environment.config.delay != 1:
        raise ValueError("feedback probing assumes delay-one token timing")
    emission = np.asarray(model.emission_matrix, dtype=np.float64)
    base = np.asarray(model.transition_matrix, dtype=np.float64)
    initial = np.asarray(model.initial_distribution, dtype=np.float64)
    return FeedbackFilters(
        emission=emission,
        initial_belief=initial,
        blind_transition=base,
        joint_emission=joint_emission(emission),
        joint_initial=joint_initial_distribution(initial),
        joint_transitions=joint_transitions(feedback_strength, base=base),
    )


def _initial_state(module: Any, batch_size: int, device: torch.device):
    state = module.get_initial_state()
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        .to(device)
        for key, value in state.items()
    }


def _policy_generator(
    randomness: PolicyRandomness,
    device: torch.device,
) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_sequence_to_int(randomness.seed_sequence, bits=64))
    return generator


@torch.no_grad()
def collect_feedback_probe_data(
    module: Any,
    env_factory,
    filters: FeedbackFilters,
    *,
    n_steps: int,
    seed: int,
    policy_mode: str = "greedy",
    n_envs: int = 16,
    device: str | torch.device = "cpu",
    warmup: int = 64,
) -> FeedbackProbeData:
    """Collect activations with every action-awareness target aligned to them."""

    if policy_mode not in {"policy", "random", "greedy"}:
        raise ValueError(f"unsupported policy mode {policy_mode!r}")
    device = torch.device(device)
    module = module.to(device).eval()
    stateful = module.is_stateful()
    if not module.heads.is_discrete:
        raise TypeError("the feedback token-guess policy must be discrete")

    emission = filters.emission
    executed = np.tile(filters.initial_belief, (n_envs, 1))
    blind = executed.copy()
    marginal = (
        None
        if filters.marginal_transition is None
        else executed.copy()
    )
    joint = np.tile(filters.joint_initial, (n_envs, 1))
    previous_tokens = np.full(n_envs, -1, dtype=np.int64)
    previous_actions = np.full(n_envs, -1, dtype=np.int64)
    generator: torch.Generator | None = None

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices: np.ndarray):
        fresh = _initial_state(module, len(indices), device)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index_tensor, fresh[key])
        return state

    def step_adapter(observations, state, randomness, action_spaces):
        nonlocal generator
        observation_tensor = torch.from_numpy(observations).float().to(device)
        if stateful:
            embedding, state = module.encode_step(observation_tensor, state)
        else:
            embedding, _ = module.encode_step(observation_tensor)
        if policy_mode == "random":
            env_actions = np.asarray(
                [space.sample() for space in action_spaces],
                dtype=np.int64,
            )
        else:
            logits = module.action_distribution_inputs(embedding)
            if policy_mode == "greedy":
                env_actions = logits.argmax(dim=-1).cpu().numpy()
            else:
                if generator is None:
                    generator = _policy_generator(randomness, device)
                probabilities = torch.softmax(logits, dim=-1)
                if torch.device(generator.device).type != probabilities.device.type:
                    probabilities = probabilities.cpu()
                env_actions = (
                    torch.multinomial(
                        probabilities,
                        1,
                        replacement=True,
                        generator=generator,
                    )
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
        return env_actions, state, embedding.cpu().numpy()

    def target_adapter(observations, infos, episode_steps):
        del observations
        for index, (info, episode_step) in enumerate(zip(infos, episode_steps)):
            if episode_step == 0:
                executed[index] = filters.initial_belief
                blind[index] = filters.initial_belief
                joint[index] = filters.joint_initial
                if marginal is not None:
                    marginal[index] = filters.initial_belief
                continue
            token = info.get("visible_token_current")
            if token is None:
                raise ValueError("a transducer update requires a visible token")
            token = int(token)
            action = int(info["executed_action"])
            transition = np.asarray(
                info["executed_transition_matrix"],
                dtype=np.float64,
            )
            measurement = np.diag(emission[:, token])
            executed[index] = predictive_belief_update(
                executed[index],
                measurement @ transition,
            )
            blind[index] = predictive_belief_update(
                blind[index],
                measurement @ filters.blind_transition,
            )
            if marginal is not None:
                marginal[index] = predictive_belief_update(
                    marginal[index],
                    measurement @ filters.marginal_transition,
                )
            joint[index] = predictive_belief_update(
                joint[index],
                np.diag(filters.joint_emission[:, token])
                @ filters.joint_transitions[action],
            )

        tokens = np.asarray(
            [
                -1
                if info.get("visible_token_current") is None
                else int(info["visible_token_current"])
                for info in infos
            ],
            dtype=np.int64,
        )
        executed_actions = np.asarray(
            [int(info.get("executed_action", -1)) for info in infos],
            dtype=np.int64,
        )
        chain, register = factor_marginals(joint)
        targets = {
            "executed": executed.copy(),
            "blind": blind.copy(),
            "joint": joint.copy(),
            "factor_m": chain,
            "factor_phi": register,
            "diagnostic": np.stack([info["belief_current"] for info in infos]),
            "tokens": tokens,
            "previous_tokens": np.where(episode_steps == 0, -1, previous_tokens),
            "previous_actions": executed_actions,
            "states": np.asarray(
                [info["state_current"] for info in infos],
                dtype=np.int64,
            ),
            "env_indices": np.arange(n_envs, dtype=np.int64),
            "episode_steps": np.asarray(episode_steps, dtype=np.int64),
        }
        if marginal is not None:
            targets["marginal"] = marginal.copy()
        previous_tokens[:] = tokens
        previous_actions[:] = executed_actions
        return targets

    collected = collect_batched_rollout_data(
        env_factory,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=n_envs,
        initial_state=initial_state if stateful else None,
        reset_state=reset_state if stateful else None,
        warmup=warmup,
    )

    def stacked(key: str, dtype=np.float64) -> np.ndarray:
        return np.asarray(collected.targets[key], dtype=dtype)

    return FeedbackProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64),
        executed=stacked("executed"),
        diagnostic=stacked("diagnostic"),
        blind=stacked("blind"),
        marginal=(
            stacked("marginal") if "marginal" in collected.targets else None
        ),
        joint=stacked("joint"),
        factor_m=stacked("factor_m"),
        factor_phi=stacked("factor_phi"),
        tokens=stacked("tokens", np.int64),
        previous_tokens=stacked("previous_tokens", np.int64),
        actions=np.asarray(collected.actions, dtype=np.int64).reshape(-1),
        previous_actions=stacked("previous_actions", np.int64),
        states=stacked("states", np.int64),
        env_indices=stacked("env_indices", np.int64),
        episode_steps=stacked("episode_steps", np.int64),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
    )


def branch_keys(data: FeedbackProbeData, depth: int = 2) -> np.ndarray:
    """Group samples by the most recent visible tokens and executed guesses."""

    token = np.where(data.tokens < 0, 3, data.tokens)
    guess = np.where(data.previous_actions < 0, 3, data.previous_actions)
    current = token * 4 + guess
    if depth == 1:
        return current
    if depth != 2:
        raise ValueError("feedback branch depth must be one or two")
    previous = np.where(data.previous_tokens < 0, 3, data.previous_tokens)
    return current * 4 + previous


def state_conditioned_guess_counts(
    data: FeedbackProbeData,
    *,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    """Count decision-time ``(state, guess)`` pairs for the marginal kernel."""

    counts = np.zeros((n_states, n_actions), dtype=np.float64)
    np.add.at(counts, (data.states, data.actions), 1.0)
    return counts
