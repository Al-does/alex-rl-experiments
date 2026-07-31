"""Multi-target transducer probe data for the composed feedback generator.

One rollout yields one activation matrix and several aligned Bayesian targets
that differ in how much of the agent's own influence, and how much of the
composed structure, they account for:

``joint``
    the exact guess-conditioned predictive belief over ``(m, phi)``; the
    transducer target of ``rl-harness``, ``K(t | a) = diag(P(t | m, phi)) U(a)``
    at delay one, and the minimal sufficient statistic of the process;
``blind``
    the same filter run under the uniformly guess-marginalized kernel, so it
    knows the feedback rule and that some guess happened but never conditions
    on which one. Assuming the register-inert kernel instead would be a
    strawman: a perfect register report contradicts it outright;
``marginal``
    the filter of the stacked, renormalized single HMM ``Ubar``, which knows the
    policy's guess statistics but not the realized guess;
``composite`` and ``composite_blind``
    their aggregations onto ``s = m + phi``, the three-state belief that fixes
    the reward and is directly comparable with the passive token-guess study;
``factor_m`` and ``factor_phi``
    the two factor predictive vectors whose product is the factored
    representation of Shai et al.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from analysis.probes import predictive_belief_update
from analysis.rollouts import PolicyRandomness, collect_batched_rollout_data
from experiments.mess3_feedback_cycle_1.dynamics import (
    chain_factor,
    composite_state_belief,
    composite_token,
    factor_marginals,
    joint_transitions,
    product_state,
)
from harness.seeding import seed_sequence_to_int


@dataclass(frozen=True, slots=True)
class FeedbackProbeData:
    """Aligned activations and every Bayesian target of the feedback loop."""

    activations: np.ndarray
    joint: np.ndarray
    diagnostic: np.ndarray
    blind: np.ndarray
    marginal: np.ndarray | None
    composite: np.ndarray
    composite_blind: np.ndarray
    factor_m: np.ndarray
    factor_phi: np.ndarray
    tokens: np.ndarray
    scored_tokens: np.ndarray
    previous_scored_tokens: np.ndarray
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
            "composite_product_mse": float(
                np.square(
                    self.composite - composite_state_belief(factored)
                ).mean()
            ),
            "register_entropy_nats": float(
                -(
                    self.factor_phi
                    * np.log(np.maximum(self.factor_phi, 1e-300))
                ).sum(axis=1).mean()
            ),
        }


@dataclass(frozen=True, slots=True)
class FeedbackFilters:
    """Every operator family needed to rebuild the targets during a rollout."""

    emission: np.ndarray
    initial_belief: np.ndarray
    transitions: np.ndarray
    n_guesses: int
    marginal_transition: np.ndarray | None = None

    @property
    def blind_transition(self) -> np.ndarray:
        """The kernel assumed by an observer who never reads its own guess.

        Marginalizing over a uniform guess keeps the kernel full support at
        every ``(kappa, epsilon)``, so the baseline filter is always defined.
        """

        return self.transitions.mean(axis=0)

    def with_marginal(self, transition: np.ndarray | None) -> FeedbackFilters:
        return FeedbackFilters(
            emission=self.emission,
            initial_belief=self.initial_belief,
            transitions=self.transitions,
            n_guesses=self.n_guesses,
            marginal_transition=(
                None
                if transition is None
                else np.asarray(transition, dtype=np.float64)
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
    n_guesses = int(round(np.sqrt(model.n_states)))
    chain = chain_factor(model.transition_matrix, size=n_guesses)
    return FeedbackFilters(
        emission=np.asarray(model.emission_matrix, dtype=np.float64),
        initial_belief=np.asarray(model.initial_distribution, dtype=np.float64),
        transitions=joint_transitions(
            feedback_strength,
            base=chain,
            n_actions=n_guesses,
        ),
        n_guesses=n_guesses,
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
    joint = np.tile(filters.initial_belief, (n_envs, 1))
    blind = joint.copy()
    marginal = None if filters.marginal_transition is None else joint.copy()
    previous_scored = np.full(n_envs, -1, dtype=np.int64)
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
                joint[index] = filters.initial_belief
                blind[index] = filters.initial_belief
                if marginal is not None:
                    marginal[index] = filters.initial_belief
                continue
            token = info.get("visible_token_current")
            if token is None:
                raise ValueError("a transducer update requires a visible token")
            measurement = np.diag(emission[:, int(token)])
            transition = np.asarray(
                info["executed_transition_matrix"],
                dtype=np.float64,
            )
            joint[index] = predictive_belief_update(
                joint[index],
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

        tokens = np.asarray(
            [
                -1
                if info.get("visible_token_current") is None
                else int(info["visible_token_current"])
                for info in infos
            ],
            dtype=np.int64,
        )
        scored = np.where(
            tokens < 0,
            -1,
            composite_token(tokens, size=filters.n_guesses),
        )
        chain, register = factor_marginals(joint)
        targets = {
            "joint": joint.copy(),
            "blind": blind.copy(),
            "composite": composite_state_belief(joint),
            "composite_blind": composite_state_belief(blind),
            "factor_m": chain,
            "factor_phi": register,
            "diagnostic": np.stack([info["belief_current"] for info in infos]),
            "tokens": tokens,
            "scored_tokens": scored,
            "previous_scored_tokens": np.where(
                episode_steps == 0,
                -1,
                previous_scored,
            ),
            "previous_actions": np.asarray(
                [int(info.get("executed_action", -1)) for info in infos],
                dtype=np.int64,
            ),
            "states": np.asarray(
                [info["state_current"] for info in infos],
                dtype=np.int64,
            ),
            "env_indices": np.arange(n_envs, dtype=np.int64),
            "episode_steps": np.asarray(episode_steps, dtype=np.int64),
        }
        if marginal is not None:
            targets["marginal"] = marginal.copy()
        previous_scored[:] = scored
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
        joint=stacked("joint"),
        diagnostic=stacked("diagnostic"),
        blind=stacked("blind"),
        marginal=(
            stacked("marginal") if "marginal" in collected.targets else None
        ),
        composite=stacked("composite"),
        composite_blind=stacked("composite_blind"),
        factor_m=stacked("factor_m"),
        factor_phi=stacked("factor_phi"),
        tokens=stacked("tokens", np.int64),
        scored_tokens=stacked("scored_tokens", np.int64),
        previous_scored_tokens=stacked("previous_scored_tokens", np.int64),
        actions=np.asarray(collected.actions, dtype=np.int64).reshape(-1),
        previous_actions=stacked("previous_actions", np.int64),
        states=stacked("states", np.int64),
        env_indices=stacked("env_indices", np.int64),
        episode_steps=stacked("episode_steps", np.int64),
        rewards=np.asarray(collected.rewards, dtype=np.float64),
    )


def branch_keys(data: FeedbackProbeData, depth: int = 2) -> np.ndarray:
    """Group samples by the most recent scored tokens and executed guesses."""

    token = np.where(data.scored_tokens < 0, 3, data.scored_tokens)
    guess = np.where(data.previous_actions < 0, 3, data.previous_actions)
    current = token * 4 + guess
    if depth == 1:
        return current
    if depth != 2:
        raise ValueError("feedback branch depth must be one or two")
    previous = np.where(
        data.previous_scored_tokens < 0,
        3,
        data.previous_scored_tokens,
    )
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
