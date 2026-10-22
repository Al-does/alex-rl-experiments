"""Task-intrinsic reference points for the passive MESS3 token-guess study.

Cycle 1 reported belief-probe R² and greedy token accuracy as bare numbers. Both
metrics have a task-imposed floor and ceiling that are large relative to the
differences between conditions, so a bare number cannot be read as a measure of
how much belief structure a training objective induced.

This module derives those reference points from the environment definition and
from the same probe estimator the conditions use, so that condition scores can be
reported as a fraction of the range the metric can actually move through.

Three references matter:

``bayes_accuracy``
    Greedy token accuracy of the exact Bayesian filter, as a function of how many
    past tokens it is allowed to see. One token reproduces the trivial
    "repeat the previous token" rule; the sequence converges well inside the
    64-token context the agents receive.

``raw_token_window_r2``
    Belief-probe R² for an affine probe read directly off the one-hot encoded
    last ``k`` observations. No network and no training are involved, so this is
    the score a policy earns for passing its own inputs through unchanged.

``untrained_module_r2``
    The same probe applied to a randomly initialised copy of the study's
    transformer. This controls for architecture, embedding width, and the final
    LayerNorm, none of which the raw-token probe exercises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from analysis.probes import r2_score
from envs.hmm import HMMEnv
from envs.mess3.model import passive_model
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    make_transducer_target,
)
from experiments.mess3_token_guess_cycle_1.analysis import (
    PROBE_RANK,
    fit_reduced_rank_affine,
)
from harness.seeding import named_seed_sequences, seed_sequence_to_int
from learners.models.transformer import TransformerModel

ALPHA = 0.85
WARMUP = 64
CONTEXT_LENGTHS = (1, 2, 3, 4, 6, 8, 16, 64)
BOOTSTRAP_RESAMPLES = 400

_STREAM_KEYS = {
    "reference_fit": (300,),
    "reference_test": (301,),
    "untrained_fit": (302,),
    "untrained_test": (303,),
    "bootstrap": (304,),
}


@dataclass(frozen=True, slots=True)
class TokenStream:
    """Exact predictive beliefs aligned with the tokens they predict."""

    beliefs: np.ndarray
    tokens: np.ndarray
    windows: np.ndarray


def simulate_stream(*, n_steps: int, seed: int, window: int = WARMUP) -> TokenStream:
    """Simulate passive MESS3 and record the exact predictive belief per step.

    The belief is the distribution over the state that emits the token being
    predicted, conditioned on every token revealed so far. This matches the
    ``delay=1`` transducer target the conditions probe against, where the
    filtering operator is ``diag(P(y|s)) @ T``.
    """

    if n_steps <= 0 or window <= 0:
        raise ValueError("n_steps and window must be positive")
    model = passive_model(alpha=ALPHA)
    transition = np.asarray(model.transition_matrix, dtype=np.float64)
    emission = np.asarray(model.emission_matrix, dtype=np.float64)
    initial = np.asarray(model.initial_distribution, dtype=np.float64)

    rng = np.random.default_rng(seed)
    state = int(rng.choice(len(initial), p=initial))
    belief = initial.copy()
    history: list[int] = [0] * window

    beliefs = np.empty((n_steps, len(initial)), dtype=np.float64)
    tokens = np.empty(n_steps, dtype=np.int64)
    windows = np.empty((n_steps, window), dtype=np.int64)

    for step in range(n_steps + window):
        token = int(rng.choice(emission.shape[1], p=emission[state]))
        if step >= window:
            index = step - window
            beliefs[index] = belief
            tokens[index] = token
            windows[index] = history[-window:]
        history.append(token)
        posterior = belief * emission[:, token]
        belief = (posterior / posterior.sum()) @ transition
        state = int(rng.choice(transition.shape[1], p=transition[state]))
    return TokenStream(beliefs=beliefs, tokens=tokens, windows=windows)


def _one_hot_window(windows: np.ndarray, k: int) -> np.ndarray:
    return np.eye(3, dtype=np.float64)[windows[:, -k:]].reshape(len(windows), -1)


def _probe_r2(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    weight, bias = fit_reduced_rank_affine(
        fit_features,
        fit_targets,
        rank=PROBE_RANK,
    )
    predicted = test_features @ weight + bias
    return r2_score(predicted, test_targets), predicted


def raw_token_window_r2(
    fit: TokenStream,
    test: TokenStream,
    *,
    context_lengths: tuple[int, ...] = CONTEXT_LENGTHS,
) -> dict[int, float]:
    """Score the study's probe against one-hot encoded raw observations."""

    scores: dict[int, float] = {}
    for k in context_lengths:
        score, _ = _probe_r2(
            _one_hot_window(fit.windows, k),
            fit.beliefs,
            _one_hot_window(test.windows, k),
            test.beliefs,
        )
        scores[k] = float(score)
    return scores


def bootstrap_r2_interval(
    fit: TokenStream,
    test: TokenStream,
    *,
    context_length: int,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Bound the probe's own sampling noise at the conditions' test-set size."""

    _, predicted = _probe_r2(
        _one_hot_window(fit.windows, context_length),
        fit.beliefs,
        _one_hot_window(test.windows, context_length),
        test.beliefs,
    )
    rng = np.random.default_rng(seed)
    n = len(test.beliefs)
    scores = [
        r2_score(predicted[index], test.beliefs[index])
        for index in (rng.integers(0, n, n) for _ in range(resamples))
    ]
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def bayes_accuracy_by_context(
    stream: TokenStream,
    *,
    context_lengths: tuple[int, ...] = CONTEXT_LENGTHS,
) -> dict[int, float]:
    """Greedy accuracy of an exact filter restricted to the last ``k`` tokens."""

    model = passive_model(alpha=ALPHA)
    transition = np.asarray(model.transition_matrix, dtype=np.float64)
    emission = np.asarray(model.emission_matrix, dtype=np.float64)
    initial = np.asarray(model.initial_distribution, dtype=np.float64)

    accuracies: dict[int, float] = {}
    for k in context_lengths:
        belief = np.repeat(initial[None, :], len(stream.windows), axis=0)
        for offset in range(k):
            belief = belief * emission[:, stream.windows[:, -k + offset]].T
            belief /= belief.sum(axis=1, keepdims=True)
            belief = belief @ transition
        predicted = (belief @ emission).argmax(axis=1)
        accuracies[k] = float((predicted == stream.tokens).mean())
    return accuracies


def _build_untrained_module(env_config: dict[str, Any], model_config: dict[str, Any]):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    environment = HMMEnv(env_config)
    try:
        spec = RLModuleSpec(
            module_class=TransformerModel,
            model_config=dict(model_config),
            observation_space=environment.observation_space,
            action_space=environment.action_space,
        )
        return spec.build()
    finally:
        environment.close()


def untrained_module_r2(
    *,
    env_config: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
    fit_steps: int,
    test_steps: int,
    device: str = "cpu",
) -> dict[str, float]:
    """Probe a randomly initialised copy of the study transformer.

    The module is never trained, so any score above the raw-token reference is
    attributable to the architecture rather than to a learning objective.
    """

    streams = named_seed_sequences(seed, _STREAM_KEYS)
    config = dict(env_config)
    config["diagnostics"] = {
        "state": True,
        "belief": True,
        "tokens": True,
        "transitions": True,
    }
    torch.manual_seed(seed_sequence_to_int(streams["untrained_fit"], bits=64))
    module = _build_untrained_module(config, model_config)

    def make_environment():
        return HMMEnv(config)

    environment = make_environment()
    try:
        initial_belief, outcome_operator, initial_operator = make_transducer_target(
            environment
        )
    finally:
        environment.close()

    common = {
        "module": module,
        "env_factory": make_environment,
        "policy_mode": "greedy",
        "device": device,
        "warmup": WARMUP,
        "initial_belief": initial_belief,
        "action_outcome_operator": outcome_operator,
        "initial_outcome_operator": initial_operator,
    }
    fit = collect_probe_data(
        n_steps=fit_steps, seed=streams["untrained_fit"], **common
    )
    test = collect_probe_data(
        n_steps=test_steps, seed=streams["untrained_test"], **common
    )
    score, _ = _probe_r2(fit.activations, fit.beliefs, test.activations, test.beliefs)
    return {
        "r_squared": float(score),
        "token_accuracy_greedy": float(test.rewards.mean()),
        "n_fit": int(len(fit.beliefs)),
        "n_test": int(len(test.beliefs)),
    }


def compute_references(
    *,
    seed: int,
    fit_steps: int,
    test_steps: int,
    env_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute every reference point the study needs to normalise its metrics."""

    streams = named_seed_sequences(seed, _STREAM_KEYS)
    fit = simulate_stream(
        n_steps=fit_steps,
        seed=seed_sequence_to_int(streams["reference_fit"]),
    )
    test = simulate_stream(
        n_steps=test_steps,
        seed=seed_sequence_to_int(streams["reference_test"]),
    )
    raw_r2 = raw_token_window_r2(fit, test)
    saturated = max(raw_r2)
    low, high = bootstrap_r2_interval(
        fit,
        test,
        context_length=min(8, saturated),
        seed=seed_sequence_to_int(streams["bootstrap"]),
    )
    accuracy = bayes_accuracy_by_context(test)
    references: dict[str, Any] = {
        "alpha": ALPHA,
        "seed": seed,
        "n_fit": fit_steps,
        "n_test": test_steps,
        "probe_rank": PROBE_RANK,
        "raw_token_window_r2": {str(k): v for k, v in raw_r2.items()},
        "belief_r2_floor": raw_r2[saturated],
        "belief_r2_floor_context": saturated,
        "belief_r2_probe_noise_95ci": [low, high],
        "bayes_accuracy_by_context": {str(k): v for k, v in accuracy.items()},
        "accuracy_floor_repeat_previous_token": accuracy[1],
        "accuracy_ceiling_bayes": accuracy[max(accuracy)],
    }
    if env_config is not None and model_config is not None:
        references["untrained_module"] = untrained_module_r2(
            env_config=env_config,
            model_config=model_config,
            seed=seed,
            fit_steps=fit_steps,
            test_steps=test_steps,
            device=device,
        )
    return references


def normalise(value: float, *, floor: float, ceiling: float) -> float:
    """Express a score as the fraction of the floor-to-ceiling range it covers."""

    if ceiling <= floor:
        raise ValueError("ceiling must exceed floor")
    return (value - floor) / (ceiling - floor)
