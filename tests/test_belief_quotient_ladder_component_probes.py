from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.mess3_reward_state_action_symmetry_cycle_7.component_probe_analysis import (
    component_targets,
    score_target,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.component_probe_campaign import (
    all_checkpoint_records,
)


def test_component_targets_use_manuscript_state_order() -> None:
    targets = component_targets(
        np.asarray(
            [
                [0.6, 0.1, 0.3],
                [0.2, 0.7, 0.1],
            ]
        )
    )

    np.testing.assert_allclose(targets["rho"], [[0.3], [0.1]])
    np.testing.assert_allclose(targets["delta"], [[0.5], [-0.5]])


def test_scalar_probe_reports_one_minus_r_squared() -> None:
    train_features = np.arange(32, dtype=np.float64).reshape(-1, 1)
    test_features = np.arange(32, 48, dtype=np.float64).reshape(-1, 1)
    train_target = 0.25 + 0.5 * train_features
    test_target = 0.25 + 0.5 * test_features

    report = score_target(
        train_features,
        test_features,
        train_target,
        test_target,
    )

    assert report["global_mse_ratio"] < 1e-12
    np.testing.assert_allclose(
        report["global_mse_ratio"],
        1.0 - report["r_squared"],
        atol=1e-12,
    )


def test_all_checkpoint_records_retains_seed55_iteration32() -> None:
    path = next(
        Path(
            "experiments/mess3_reward_state_action_symmetry_cycle_7/"
            "variant_1_ctx32_ent/results"
        ).glob("*-seed55-10m/checkpoint_probe_curve.json")
    )

    records = all_checkpoint_records(path)

    assert [stage for stage, _ in records] == [
        "init",
        "iteration_1",
        "iteration_2",
        "iteration_4",
        "iteration_8",
        "iteration_16",
        "iteration_32",
        "final",
    ]
    assert [record.get("training_iteration") for _, record in records] == [
        None,
        1,
        2,
        4,
        8,
        16,
        32,
        41,
    ]
