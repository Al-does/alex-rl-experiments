"""Closed-loop composition analysis for guess-driven MESS3 feedback.

Two questions live here, both answerable without a trained network:

1. What accuracy can a myopic Bayes filter reach once its own guesses steer the
   process? This is the ceiling the reinforcement-learning arms are measured
   against.
2. Once the policy is fixed, can the agent-in-the-loop process be rewritten as
   a single autonomous HMM whose transition matrix stacks the guess-conditioned
   kernels and renormalizes them? The candidate is

       Ubar[s, .] = sum_y P(a = y | s) U(y)[s, .],

   the guess-marginalized kernel. It is exact when the guess is conditionally
   independent of the hidden state given nothing else, and only approximate
   when the policy conditions on a belief that the state itself is correlated
   with. The report below measures the residual directly, both on beliefs and
   on the distribution over observed token blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.hmm import stationary_distribution
from envs.mess3.model import PASSIVE_TRANSITION_MATRIX, emission_matrix
from experiments.mess3_feedback_cycle_1.dynamics import (
    executed_state_belief,
    factor_marginals,
    feedback_transitions,
    joint_emission,
    joint_initial_distribution,
    joint_transitions,
    product_state,
)


POLICIES = ("myopic_argmax", "probability_matching", "uniform")
DEFAULT_BLOCK_LENGTH = 4


@dataclass(frozen=True, slots=True)
class ClosedLoopRollout:
    """Exact-filter trajectories for one policy in the feedback loop."""

    states: np.ndarray
    actions: np.ndarray
    tokens: np.ndarray
    rewards: np.ndarray
    beliefs: np.ndarray
    joint_beliefs: np.ndarray | None


def _sample_categorical(
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=-1)
    draws = rng.random((probabilities.shape[0], 1))
    index = (draws > cumulative).sum(axis=-1)
    return np.minimum(index, probabilities.shape[-1] - 1).astype(np.int64)


def _normalize(rows: np.ndarray) -> np.ndarray:
    total = rows.sum(axis=-1, keepdims=True)
    return rows / np.maximum(total, 1e-300)


def _choose_actions(
    predictive: np.ndarray,
    policy: str,
    rng: np.random.Generator,
    n_actions: int,
) -> np.ndarray:
    if policy == "myopic_argmax":
        return predictive.argmax(axis=-1).astype(np.int64)
    if policy == "probability_matching":
        return _sample_categorical(rng, predictive)
    if policy == "uniform":
        return rng.integers(0, n_actions, size=len(predictive)).astype(np.int64)
    raise ValueError(
        f"closed-loop policy must be one of {POLICIES}, got {policy!r}"
    )


def simulate_closed_loop(
    strength: float,
    *,
    policy: str = "myopic_argmax",
    n_chains: int = 256,
    n_steps: int = 2_048,
    burn_in: int = 64,
    seed: int = 0,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
    context_length: int | None = None,
    record_joint: bool = False,
) -> ClosedLoopRollout:
    """Run the feedback loop with an exact or window-truncated Bayes filter.

    ``context_length`` restarts the acting filter from the stationary prior
    that many decisions back, matching what a transformer with that many
    ``(token, guess)`` observations can compute.
    """

    if n_chains <= 0 or n_steps <= 0:
        raise ValueError("n_chains and n_steps must be positive")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must be non-negative and shorter than n_steps")
    if context_length is not None and context_length <= 0:
        raise ValueError("context_length must be positive when supplied")

    rng = np.random.default_rng(seed)
    transitions = feedback_transitions(strength, base=base)
    emission = emission_matrix(alpha)
    n_states = emission.shape[0]
    n_actions = transitions.shape[0]
    prior = stationary_distribution(base)

    state = _sample_categorical(rng, np.tile(prior, (n_chains, 1)))
    token = _sample_categorical(rng, emission[state])
    exact = np.tile(prior, (n_chains, 1))
    window = (
        None
        if context_length is None
        else np.tile(np.eye(n_states), (n_chains, context_length, 1, 1))
    )
    joint = (
        np.tile(joint_initial_distribution(prior), (n_chains, 1))
        if record_joint
        else None
    )
    joint_kernels = joint_transitions(strength, base=base) if record_joint else None
    joint_likelihood = joint_emission(emission) if record_joint else None

    kept = n_steps - burn_in
    states = np.empty((n_chains, kept), dtype=np.int64)
    actions = np.empty((n_chains, kept), dtype=np.int64)
    tokens = np.empty((n_chains, kept), dtype=np.int64)
    rewards = np.empty((n_chains, kept), dtype=np.float64)
    beliefs = np.empty((n_chains, kept, n_states), dtype=np.float64)
    joint_beliefs = (
        np.empty((n_chains, kept, n_states * n_states), dtype=np.float64)
        if record_joint
        else None
    )

    for step in range(n_steps):
        if window is None:
            acting = exact
        else:
            acting = np.tile(prior, (n_chains, 1))
            for slot in range(context_length):
                acting = _normalize(
                    np.einsum("ni,nij->nj", acting, window[:, slot])
                )
        predictive = acting @ emission
        action = _choose_actions(predictive, policy, rng, n_actions)
        reward = (action == token).astype(np.float64)

        if step >= burn_in:
            index = step - burn_in
            states[:, index] = state
            actions[:, index] = action
            tokens[:, index] = token
            rewards[:, index] = reward
            beliefs[:, index] = acting
            if joint_beliefs is not None:
                joint_beliefs[:, index] = joint

        kernel = emission[:, token].T[:, :, None] * transitions[action]
        exact = _normalize(np.einsum("ni,nij->nj", exact, kernel))
        if window is not None:
            window[:, :-1] = window[:, 1:]
            window[:, -1] = kernel
        if joint is not None:
            joint_kernel = (
                joint_likelihood[:, token].T[:, :, None]
                * joint_kernels[action]
            )
            joint = _normalize(np.einsum("ni,nij->nj", joint, joint_kernel))

        state = _sample_categorical(rng, transitions[action][np.arange(n_chains), state])
        token = _sample_categorical(rng, emission[state])

    return ClosedLoopRollout(
        states=states,
        actions=actions,
        tokens=tokens,
        rewards=rewards,
        beliefs=beliefs,
        joint_beliefs=joint_beliefs,
    )


def myopic_ceiling(
    strength: float,
    *,
    context_length: int | None = None,
    n_chains: int = 256,
    n_steps: int = 1_024,
    burn_in: int = 32,
    seed: int = 1,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> dict[str, float]:
    """Estimate the accuracy of the pointwise-optimal gamma-zero policy.

    Under ``gamma = 0`` the policy gradient gives no credit for how a guess
    reshapes future beliefs, so the fixed point of policy improvement is the
    myopic argmax of the current predictive distribution. That is the ceiling
    the trained arms are measured against.
    """

    rollout = simulate_closed_loop(
        strength,
        policy="myopic_argmax",
        n_chains=n_chains,
        n_steps=n_steps,
        burn_in=burn_in,
        seed=seed,
        alpha=alpha,
        base=base,
        context_length=context_length,
    )
    per_chain = rollout.rewards.mean(axis=1)
    return {
        "accuracy": float(per_chain.mean()),
        "stderr": float(per_chain.std(ddof=1) / np.sqrt(len(per_chain))),
        "context_length": context_length,
        "n_samples": int(rollout.rewards.size),
    }


def marginalized_transition(
    states: np.ndarray,
    actions: np.ndarray,
    transitions: np.ndarray,
    *,
    prior_count: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the guess-conditioned kernels by the state-conditioned guess law.

    Returns ``(guess_given_state, marginal_transition)``. Laplace smoothing
    keeps the estimate well defined for states the policy rarely visits.
    """

    flat_states = np.asarray(states, dtype=np.int64).reshape(-1)
    flat_actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if flat_states.shape != flat_actions.shape:
        raise ValueError("states and actions must be aligned")
    n_actions, n_states, _ = np.asarray(transitions).shape
    counts = np.full((n_states, n_actions), float(prior_count))
    np.add.at(counts, (flat_states, flat_actions), 1.0)
    guess_given_state = counts / counts.sum(axis=1, keepdims=True)
    marginal = np.einsum(
        "sa,asj->sj",
        guess_given_state,
        np.asarray(transitions, dtype=np.float64),
    )
    return guess_given_state, marginal


def hmm_filter(
    tokens: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
    *,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Filter an autonomous HMM over recorded delay-one token streams."""

    observations = np.asarray(tokens, dtype=np.int64)
    if observations.ndim != 2:
        raise ValueError("tokens must have shape (chains, steps)")
    kernel = np.asarray(transition, dtype=np.float64)
    likelihood = np.asarray(emission, dtype=np.float64)
    prior = (
        stationary_distribution(kernel) if initial is None else np.asarray(initial)
    )
    n_chains, n_steps = observations.shape
    beliefs = np.empty((n_chains, n_steps, kernel.shape[0]), dtype=np.float64)
    belief = np.tile(prior, (n_chains, 1))
    for step in range(n_steps):
        beliefs[:, step] = belief
        measured = belief * likelihood[:, observations[:, step]].T
        belief = _normalize(measured @ kernel)
    return beliefs


def block_distribution(
    transition: np.ndarray,
    emission: np.ndarray,
    *,
    length: int,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Return the exact stationary distribution over token blocks.

    Blocks are indexed little-endian in time: the first token is the least
    significant digit, matching :func:`empirical_block_distribution`.
    """

    if length <= 0:
        raise ValueError("block length must be positive")
    kernel = np.asarray(transition, dtype=np.float64)
    likelihood = np.asarray(emission, dtype=np.float64)
    prior = (
        stationary_distribution(kernel) if initial is None else np.asarray(initial)
    )
    weighted = prior.reshape(1, -1)
    for _ in range(length):
        weighted = np.concatenate(
            [
                (weighted * likelihood[:, token]) @ kernel
                for token in range(likelihood.shape[1])
            ],
            axis=0,
        )
    return weighted.sum(axis=1)


def empirical_block_distribution(
    tokens: np.ndarray,
    *,
    length: int,
    n_tokens: int,
) -> np.ndarray:
    """Count token blocks with the little-endian index used above."""

    observations = np.asarray(tokens, dtype=np.int64)
    if observations.ndim != 2:
        raise ValueError("tokens must have shape (chains, steps)")
    if observations.shape[1] <= length:
        raise ValueError("each chain must be longer than one block")
    usable = observations.shape[1] - length + 1
    index = np.zeros((observations.shape[0], usable), dtype=np.int64)
    for offset in range(length):
        index += observations[:, offset : offset + usable] * (n_tokens**offset)
    counts = np.bincount(index.reshape(-1), minlength=n_tokens**length)
    return counts / counts.sum()


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Return the total-variation distance between two distributions."""

    return float(0.5 * np.abs(np.asarray(left) - np.asarray(right)).sum())


def factorization_report(rollout: ClosedLoopRollout) -> dict[str, float]:
    """Measure how much the product-state approximation loses."""

    if rollout.joint_beliefs is None:
        raise ValueError("factorization analysis requires recorded joint beliefs")
    joint = rollout.joint_beliefs.reshape(-1, rollout.joint_beliefs.shape[-1])
    chain, register = factor_marginals(joint)
    factored = product_state(chain, register)
    executed = executed_state_belief(joint)
    factored_executed = executed_state_belief(factored)
    return {
        "joint_product_mse": float(np.square(joint - factored).mean()),
        "executed_product_mse": float(
            np.square(executed - factored_executed).mean()
        ),
        "executed_product_max_abs": float(
            np.abs(executed - factored_executed).max()
        ),
        "register_entropy_nats": float(
            -(register * np.log(np.maximum(register, 1e-300))).sum(axis=1).mean()
        ),
        "register_max_probability": float(register.max(axis=1).mean()),
    }


def single_hmm_report(
    strength: float,
    *,
    policy: str = "myopic_argmax",
    n_chains: int = 256,
    n_steps: int = 2_048,
    burn_in: int = 64,
    seed: int = 0,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    context_length: int | None = None,
    record_joint: bool = True,
) -> dict[str, Any]:
    """Test whether one stacked, renormalized HMM reproduces the closed loop."""

    rollout = simulate_closed_loop(
        strength,
        policy=policy,
        n_chains=n_chains,
        n_steps=n_steps,
        burn_in=burn_in,
        seed=seed,
        alpha=alpha,
        base=base,
        context_length=context_length,
        record_joint=record_joint,
    )
    emission = emission_matrix(alpha)
    transitions = feedback_transitions(strength, base=base)
    guess_given_state, marginal = marginalized_transition(
        rollout.states,
        rollout.actions,
        transitions,
    )

    n_tokens = emission.shape[1]
    empirical = empirical_block_distribution(
        rollout.tokens,
        length=block_length,
        n_tokens=n_tokens,
    )
    half = max(1, rollout.tokens.shape[0] // 2)
    sampling_floor = total_variation(
        empirical_block_distribution(
            rollout.tokens[:half],
            length=block_length,
            n_tokens=n_tokens,
        ),
        empirical_block_distribution(
            rollout.tokens[half:],
            length=block_length,
            n_tokens=n_tokens,
        ),
    )
    marginal_blocks = block_distribution(
        marginal,
        emission,
        length=block_length,
    )
    base_blocks = block_distribution(base, emission, length=block_length)

    marginal_beliefs = hmm_filter(rollout.tokens, marginal, emission)
    exact_beliefs = rollout.beliefs
    belief_mse = float(np.square(marginal_beliefs - exact_beliefs).mean())

    report: dict[str, Any] = {
        "feedback_strength": float(strength),
        "policy": policy,
        "context_length": context_length,
        "block_length": int(block_length),
        "n_samples": int(rollout.tokens.size),
        "myopic_accuracy": float(rollout.rewards.mean()),
        "myopic_accuracy_stderr": float(
            rollout.rewards.mean(axis=1).std(ddof=1) / np.sqrt(len(rollout.rewards))
        ),
        "guess_given_state": guess_given_state.tolist(),
        "marginal_transition": marginal.tolist(),
        "marginal_stationary": stationary_distribution(marginal).tolist(),
        "block_tv_marginal_hmm": total_variation(empirical, marginal_blocks),
        "block_tv_base_hmm": total_variation(empirical, base_blocks),
        "block_tv_sampling_floor": sampling_floor,
        "belief_mse_marginal_vs_exact": belief_mse,
        "belief_variance_exact": float(
            np.square(exact_beliefs - exact_beliefs.mean(axis=(0, 1))).mean()
        ),
    }
    report["single_hmm_excess_tv"] = (
        report["block_tv_marginal_hmm"] - sampling_floor
    )
    if record_joint:
        report.update(factorization_report(rollout))
    return report
