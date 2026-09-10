from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from envs.hmm import HMMModel, stationary_distribution


COMPONENT_PARAMETERS = (
    {"name": "mess3_a", "x": 0.15, "alpha": 0.60},
    {"name": "mess3_b", "x": 0.50, "alpha": 0.66},
)
COMPONENT_COUNT = 2
STATES_PER_COMPONENT = 3
STATE_COUNT = COMPONENT_COUNT * STATES_PER_COMPONENT
TOKEN_COUNT = 3
CONTEXT_LENGTH = 128
EPISODE_LENGTH = 127


def mess3_edge_matrices(*, x: float, alpha: float) -> np.ndarray:
    if not 0.0 <= x <= 0.5:
        raise ValueError("x must lie in [0, 0.5]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    beta = (1.0 - alpha) / 2.0
    y = 1.0 - 2.0 * x
    return np.asarray(
        [
            [
                [alpha * y, beta * x, beta * x],
                [alpha * x, beta * y, beta * x],
                [alpha * x, beta * x, beta * y],
            ],
            [
                [beta * y, alpha * x, beta * x],
                [beta * x, alpha * y, beta * x],
                [beta * x, alpha * x, beta * y],
            ],
            [
                [beta * y, beta * x, alpha * x],
                [beta * x, beta * y, alpha * x],
                [beta * x, beta * x, alpha * y],
            ],
        ],
        dtype=np.float64,
    )


def nonergodic_mess3_model(
    *,
    component_prior: Sequence[float] = (0.5, 0.5),
) -> HMMModel:
    prior = np.asarray(component_prior, dtype=np.float64)
    if prior.shape != (COMPONENT_COUNT,):
        raise ValueError("component_prior must contain two probabilities")
    if (prior < 0.0).any() or not np.isclose(prior.sum(), 1.0):
        raise ValueError("component_prior must be a probability distribution")

    edges = np.zeros((TOKEN_COUNT, STATE_COUNT, STATE_COUNT), dtype=np.float64)
    initial = np.zeros(STATE_COUNT, dtype=np.float64)
    for component, parameters in enumerate(COMPONENT_PARAMETERS):
        start = component * STATES_PER_COMPONENT
        stop = start + STATES_PER_COMPONENT
        component_edges = mess3_edge_matrices(
            x=float(parameters["x"]),
            alpha=float(parameters["alpha"]),
        )
        edges[:, start:stop, start:stop] = component_edges
        transition = component_edges.sum(axis=0)
        initial[start:stop] = prior[component] * stationary_distribution(transition)

    transition = edges.sum(axis=0)
    emission = edges.sum(axis=2).T
    return HMMModel(
        initial_distribution=initial,
        transition_matrix=transition,
        emission_matrix=emission,
        edge_transition_matrices=edges,
        state_labels=tuple(
            f"{parameters['name']}_state_{state}"
            for parameters in COMPONENT_PARAMETERS
            for state in range(STATES_PER_COMPONENT)
        ),
        token_labels=("a", "b", "c"),
    )


def environment_config() -> dict[str, object]:
    return {
        "model": {
            "factory": (
                "experiments.nonergodic_mess3_token_guess_cycle_1.process:"
                "nonergodic_mess3_model"
            ),
        },
        "task": {
            "class": (
                "experiments.nonergodic_mess3_token_guess_cycle_1.task:"
                "NextTokenGuessTask"
            ),
        },
        "observation": {
            "token": {"depth": 1},
            "action": None,
        },
        "delay": 1,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": False,
    }
