from __future__ import annotations

import numpy as np

from envs.strata.model import strata_model
from experiments.strata_context_64_entropy_4x_posthoc.analysis import (
    _probability_rows,
    _token_ntp,
)
from experiments.strata_context_64_entropy_4x_posthoc.report import (
    _selected_archived,
)


def test_probability_rows_removes_roundoff_without_changing_distribution():
    values = np.asarray(
        [
            [0.0, 0.25, 0.7500000000000002],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    normalized = _probability_rows(values)
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
    np.testing.assert_allclose(normalized[1], values[1])


def test_delay_one_ntp_recovers_pending_token_distribution():
    model = strata_model(alpha=0.98, t0=0.30, t1=0.80)
    source = np.asarray(
        [
            [0.2, 0.3, 0.5],
            [0.7, 0.1, 0.2],
        ],
        dtype=np.float64,
    )
    arrival = source @ model.transition_matrix
    np.testing.assert_allclose(
        _token_ntp(arrival),
        source @ model.emission_matrix,
        atol=1e-12,
        rtol=0.0,
    )


def test_selected_archived_requires_exact_checkpoint_labels():
    summary = {
        "checkpoint_reports": [
            {
                "checkpoint": "initial_checkpoint",
                "agent_steps": 0,
                "training_iteration": 0,
                "policy": {"mean_reward": 0.3},
            },
            {
                "checkpoint": "iteration_1",
                "agent_steps": 10,
                "training_iteration": 1,
                "policy": {"mean_reward": 0.5},
            },
            {
                "checkpoint": "checkpoint_000000",
                "agent_steps": 20,
                "training_iteration": 2,
                "policy": {"mean_reward": 0.6},
            },
        ]
    }
    trajectory, selected = _selected_archived(
        summary,
        {
            "init": "initial_checkpoint",
            "penultimate": "iteration_1",
            "final": "checkpoint_000000",
        },
    )
    assert len(trajectory) == 3
    assert selected["penultimate"]["mean_reward"] == 0.5
