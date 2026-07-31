"""Closed-loop analysis of the composed guess-driven generator.

Three questions live here, all answerable without training a network:

1. How hard is the loop at each ``(kappa, epsilon)``? The myopic Bayes ceiling
   is what the reinforcement-learning arms are measured against.
2. What does a factored representation cost? A product state ``b_m (x) b_phi``
   is compared against the exact joint belief, in extra nats per token on the
   scored sub-token. This is the tradeoff of Shai et al. (arXiv:2602.02385).
3. Can the agent-in-the-loop process be rewritten as one autonomous HMM whose
   transition matrix stacks the guess-conditioned kernels and renormalizes
   them, ``Ubar[s, .] = sum_y P(guess = y | state = s) U(y)[s, .]``? It is exact
   when the guess is conditionally independent of the hidden state given the
   state, and only approximate when the policy conditions on a belief that the
   state is correlated with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.hmm import stationary_distribution
from envs.mess3.model import (
    N_STATES,
    PASSIVE_TRANSITION_MATRIX,
    emission_matrix,
)
from experiments.mess3_feedback_factoring_cycle_1.dynamics import (
    composite_likelihood,
    composite_state_belief,
    composite_token,
    factor_marginals,
    joint_emission,
    joint_initial_distribution,
    joint_transitions,
    product_state,
)


POLICIES = ("myopic_argmax", "probability_matching", "uniform")
DEFAULT_BLOCK_LENGTH = 4


@dataclass(frozen=True, slots=True)
class ComposedProcess:
    """Every operator family of one ``(kappa, epsilon)`` generator."""

    feedback_strength: float
    register_noise: float
    transitions: np.ndarray
    emission: np.ndarray
    scored_likelihood: np.ndarray
    initial: np.ndarray
    n_guesses: int

    @property
    def n_states(self) -> int:
        return self.transitions.shape[1]

    @property
    def window_prior(self) -> np.ndarray:
        """The mid-stream prior a finite-context observer must start from.

        Episodes begin with the register pinned at zero, but a window that
        opens mid-episode knows nothing about it. Both factors are
        uniform-stationary, so the uninformative product prior is uniform over
        the whole state product; asserting ``phi = 0`` instead would be
        contradicted outright by an accurate register report.
        """

        return np.full(self.n_states, 1.0 / self.n_states)

    def predictive(self, belief: np.ndarray) -> np.ndarray:
        """Return the distribution over the next scored sub-token."""

        return np.asarray(belief) @ self.scored_likelihood


def composed_process(
    feedback_strength: float,
    register_noise: float,
    *,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> ComposedProcess:
    """Assemble the composed generator used by both theory and probing."""

    likelihood = emission_matrix(alpha)
    return ComposedProcess(
        feedback_strength=float(feedback_strength),
        register_noise=float(register_noise),
        transitions=joint_transitions(feedback_strength, base=base),
        emission=joint_emission(likelihood, register_noise=register_noise),
        scored_likelihood=composite_likelihood(
            likelihood,
            register_noise=register_noise,
        ),
        initial=joint_initial_distribution(stationary_distribution(base)),
        n_guesses=N_STATES,
    )


@dataclass(frozen=True, slots=True)
class ClosedLoopRollout:
    """Exact-filter trajectories for one policy in the feedback loop."""

    states: np.ndarray
    actions: np.ndarray
    tokens: np.ndarray
    scored_tokens: np.ndarray
    rewards: np.ndarray
    beliefs: np.ndarray
    factored_costs: np.ndarray


def _sample_categorical(
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=-1)
    draws = rng.random((probabilities.shape[0], 1))
    index = (draws > cumulative).sum(axis=-1)
    return np.minimum(index, probabilities.shape[-1] - 1).astype(np.int64)


def _normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(rows.sum(axis=-1, keepdims=True), 1e-300)


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


def factored_cost(process: ComposedProcess, beliefs: np.ndarray) -> np.ndarray:
    """Extra nats per token paid for a product-state representation.

    This is the fidelity the Factored World Hypothesis trades away: the
    Kullback-Leibler divergence from the exact predictive distribution over the
    scored sub-token to the one implied by ``b_m (x) b_phi``.
    """

    exact = _normalize(process.predictive(beliefs))
    chain, register = factor_marginals(beliefs)
    factored = _normalize(process.predictive(product_state(chain, register)))
    return (
        exact * (np.log(np.maximum(exact, 1e-300)) - np.log(np.maximum(factored, 1e-300)))
    ).sum(axis=-1)


def simulate_closed_loop(
    feedback_strength: float,
    register_noise: float,
    *,
    policy: str = "myopic_argmax",
    n_chains: int = 256,
    n_steps: int = 2_048,
    burn_in: int = 128,
    seed: int = 0,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
    context_length: int | None = None,
) -> ClosedLoopRollout:
    """Run the feedback loop with an exact or window-truncated Bayes filter.

    ``context_length`` restarts the acting filter from the stationary prior that
    many decisions back, matching what a transformer with that many
    ``(token, guess)`` observations can compute.
    """

    if n_chains <= 0 or n_steps <= 0:
        raise ValueError("n_chains and n_steps must be positive")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must be non-negative and shorter than n_steps")
    if context_length is not None and context_length <= 0:
        raise ValueError("context_length must be positive when supplied")

    rng = np.random.default_rng(seed)
    process = composed_process(
        feedback_strength,
        register_noise,
        alpha=alpha,
        base=base,
    )
    n_states = process.n_states
    prior = process.initial

    state = _sample_categorical(rng, np.tile(prior, (n_chains, 1)))
    token = _sample_categorical(rng, process.emission[state])
    exact = np.tile(prior, (n_chains, 1))
    window = (
        None
        if context_length is None
        else np.tile(np.eye(n_states), (n_chains, context_length, 1, 1))
    )

    kept = n_steps - burn_in
    states = np.empty((n_chains, kept), dtype=np.int64)
    actions = np.empty((n_chains, kept), dtype=np.int64)
    tokens = np.empty((n_chains, kept), dtype=np.int64)
    rewards = np.empty((n_chains, kept), dtype=np.float64)
    beliefs = np.empty((n_chains, kept, n_states), dtype=np.float64)

    for step in range(n_steps):
        if window is None:
            acting = exact
        else:
            acting = np.tile(process.window_prior, (n_chains, 1))
            for slot in range(context_length):
                acting = _normalize(
                    np.einsum("ni,nij->nj", acting, window[:, slot])
                )
        action = _choose_actions(
            process.predictive(acting),
            policy,
            rng,
            process.n_guesses,
        )
        scored = composite_token(token, size=process.n_guesses)

        if step >= burn_in:
            index = step - burn_in
            states[:, index] = state
            actions[:, index] = action
            tokens[:, index] = token
            rewards[:, index] = (action == scored).astype(np.float64)
            beliefs[:, index] = acting

        kernel = (
            process.emission[:, token].T[:, :, None] * process.transitions[action]
        )
        exact = _normalize(np.einsum("ni,nij->nj", exact, kernel))
        if window is not None:
            window[:, :-1] = window[:, 1:]
            window[:, -1] = kernel
        state = _sample_categorical(
            rng,
            process.transitions[action][np.arange(n_chains), state],
        )
        token = _sample_categorical(rng, process.emission[state])

    return ClosedLoopRollout(
        states=states,
        actions=actions,
        tokens=tokens,
        scored_tokens=composite_token(tokens, size=process.n_guesses),
        rewards=rewards,
        beliefs=beliefs,
        factored_costs=factored_cost(
            process,
            beliefs.reshape(-1, n_states),
        ).reshape(n_chains, kept),
    )


def myopic_ceiling(
    feedback_strength: float,
    register_noise: float,
    *,
    context_length: int | None = None,
    n_chains: int = 256,
    n_steps: int = 1_024,
    burn_in: int = 64,
    seed: int = 1,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
) -> dict[str, float]:
    """Estimate the accuracy of the pointwise-optimal gamma-zero policy.

    Under ``gamma = 0`` the policy gradient gives no credit for how a guess
    reshapes future beliefs, so the fixed point of policy improvement is the
    myopic argmax of the current predictive distribution.
    """

    rollout = simulate_closed_loop(
        feedback_strength,
        register_noise,
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
        belief = _normalize(
            (belief * likelihood[:, observations[:, step]].T) @ kernel
        )
    return beliefs


def block_distribution(
    transition: np.ndarray,
    emission: np.ndarray,
    *,
    length: int,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Return the exact stationary distribution over scored-token blocks.

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


def factorization_report(
    process: ComposedProcess,
    rollout: ClosedLoopRollout,
) -> dict[str, float]:
    """Measure how much the product-state approximation loses."""

    joint = rollout.beliefs.reshape(-1, process.n_states)
    chain, register = factor_marginals(joint)
    factored = product_state(chain, register)
    composite = composite_state_belief(joint)
    return {
        "factored_cost_nats": float(rollout.factored_costs.mean()),
        "joint_product_mse": float(np.square(joint - factored).mean()),
        "composite_product_mse": float(
            np.square(composite - composite_state_belief(factored)).mean()
        ),
        "chain_entropy_nats": float(
            -(chain * np.log(np.maximum(chain, 1e-300))).sum(axis=1).mean()
        ),
        "register_entropy_nats": float(
            -(register * np.log(np.maximum(register, 1e-300))).sum(axis=1).mean()
        ),
    }


def single_hmm_report(
    feedback_strength: float,
    register_noise: float,
    *,
    policy: str = "myopic_argmax",
    n_chains: int = 256,
    n_steps: int = 2_048,
    burn_in: int = 128,
    seed: int = 0,
    alpha: float = 0.85,
    base: np.ndarray = PASSIVE_TRANSITION_MATRIX,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Test whether one stacked, renormalized HMM reproduces the closed loop."""

    process = composed_process(
        feedback_strength,
        register_noise,
        alpha=alpha,
        base=base,
    )
    rollout = simulate_closed_loop(
        feedback_strength,
        register_noise,
        policy=policy,
        n_chains=n_chains,
        n_steps=n_steps,
        burn_in=burn_in,
        seed=seed,
        alpha=alpha,
        base=base,
        context_length=context_length,
    )
    guess_given_state, marginal = marginalized_transition(
        rollout.states,
        rollout.actions,
        process.transitions,
    )

    n_scored = process.n_guesses
    empirical = empirical_block_distribution(
        rollout.scored_tokens,
        length=block_length,
        n_tokens=n_scored,
    )
    half = max(1, rollout.scored_tokens.shape[0] // 2)
    sampling_floor = total_variation(
        empirical_block_distribution(
            rollout.scored_tokens[:half],
            length=block_length,
            n_tokens=n_scored,
        ),
        empirical_block_distribution(
            rollout.scored_tokens[half:],
            length=block_length,
            n_tokens=n_scored,
        ),
    )
    marginal_blocks = block_distribution(
        marginal,
        process.scored_likelihood,
        length=block_length,
    )
    inert_blocks = block_distribution(
        process.transitions[0],
        process.scored_likelihood,
        length=block_length,
    )
    marginal_beliefs = hmm_filter(rollout.tokens, marginal, process.emission)

    report: dict[str, Any] = {
        "feedback_strength": float(feedback_strength),
        "register_noise": float(register_noise),
        "policy": policy,
        "context_length": context_length,
        "block_length": int(block_length),
        "n_samples": int(rollout.scored_tokens.size),
        "myopic_accuracy": float(rollout.rewards.mean()),
        "myopic_accuracy_stderr": float(
            rollout.rewards.mean(axis=1).std(ddof=1) / np.sqrt(len(rollout.rewards))
        ),
        "guess_given_state": guess_given_state.tolist(),
        "marginal_transition": marginal.tolist(),
        "block_tv_marginal_hmm": total_variation(empirical, marginal_blocks),
        "block_tv_inert_hmm": total_variation(empirical, inert_blocks),
        "block_tv_sampling_floor": sampling_floor,
        "belief_mse_marginal_vs_exact": float(
            np.square(marginal_beliefs - rollout.beliefs).mean()
        ),
        **factorization_report(process, rollout),
    }
    report["single_hmm_excess_tv"] = (
        report["block_tv_marginal_hmm"] - sampling_floor
    )
    return report
