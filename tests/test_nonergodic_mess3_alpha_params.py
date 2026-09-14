from __future__ import annotations

import numpy as np

from experiments.nonergodic_mess3_supervised.alpha_params.process import (
    COMPONENT_PARAMETERS,
    STATES_PER_COMPONENT,
    nonergodic_mess3_model,
)


def test_alpha_params_component_parameters():
    assert COMPONENT_PARAMETERS == (
        {"name": "mess3_a", "x": 0.20, "alpha": 0.95},
        {"name": "mess3_b", "x": 0.4667, "alpha": 0.95},
    )
    model = nonergodic_mess3_model()
    np.testing.assert_allclose(
        model.initial_distribution.reshape(2, STATES_PER_COMPONENT).sum(axis=1),
        [0.5, 0.5],
    )
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0)
