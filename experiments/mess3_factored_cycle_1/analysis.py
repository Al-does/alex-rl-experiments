"""Experiment-facing probe targets and PR 35 geometry reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from analysis.probes import (
    regression_factor_geometry,
    representation_dimension_predictions,
    variance_geometry,
)
from experiments.mess3_factored_cycle_1.reference import factor_targets


def _contrasts(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return values[..., :2] - values[..., 2:3]


def geometry_report(
    activations: np.ndarray,
    joint_beliefs: np.ndarray,
    *,
    expected_quotient_dimension: int | None = None,
) -> dict[str, Any]:
    """Measure CEV and factor readout geometry with PR 35's public API.

    Factor subspaces are identified by one joint affine regression to centered
    two-coordinate simplex contrasts. Interpretation still requires held-out
    decodability; this function intentionally does not label a factor present
    or absent from training data alone.
    """

    targets = factor_targets(joint_beliefs)
    factor_geometry = regression_factor_geometry(
        np.asarray(activations, dtype=np.float64),
        {
            "f1": _contrasts(targets["f1"]),
            "f2": _contrasts(targets["f2"]),
        },
        target_ranks={"f1": 2, "f2": 2},
    )
    report: dict[str, Any] = {
        "activation_geometry": variance_geometry(
            np.asarray(activations, dtype=np.float64),
            max_spectrum_entries=32,
        ),
        "factor_geometry": factor_geometry,
        "dimension_predictions": representation_dimension_predictions((3, 3)),
        "joint_product_mse": float(
            np.square(targets["joint_residual"]).mean()
        ),
        "interpretation": (
            "Geometry is descriptive until held-out probes establish that "
            "each corresponding belief target is decodable."
        ),
    }
    if expected_quotient_dimension is not None:
        report["expected_reward_quotient_dimension"] = int(
            expected_quotient_dimension
        )
    return report


def nested_function_features(
    first_belief: np.ndarray,
    second_belief: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Return E3's factor-only and factor-plus-interaction readout features."""

    first = _contrasts(first_belief)
    second = _contrasts(second_belief)
    factor_only = np.concatenate(
        [np.ones((len(first), 1)), first, second],
        axis=1,
    )
    interactions = (
        first[:, :, None] * second[:, None, :]
    ).reshape(len(first), -1)
    return {
        "factor_only": factor_only,
        "with_joint_interactions": np.concatenate(
            [factor_only, interactions],
            axis=1,
        ),
    }
