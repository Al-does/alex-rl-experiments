"""Tests for campaign-0040 longitudinal belief-symmetry probes."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes_0040.analysis import (
    CHECKPOINT_LABELS,
    PROBE_ITERATIONS,
    _checkpoint_name_for_iteration,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes_0040.seed_queue import (
    CAMPAIGN_SUFFIX,
    _required_tune_checkpoints,
)


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (2, "checkpoint_000001"),
        (8, "checkpoint_000007"),
        (22, "checkpoint_000021"),
    ],
)
def test_checkpoint_name_for_iteration(iteration, expected):
    assert _checkpoint_name_for_iteration(iteration) == expected


def test_required_tune_checkpoints_matches_cycle_five_layout():
    summary = {
        "trials": [
            {
                "checkpoint": "/tmp/checkpoint_000021",
                "best": "/tmp/checkpoint_000001",
                "last": {"path": "/tmp/checkpoint_000021"},
            }
        ]
    }
    # Include intermediate checkpoint paths so _walk_strings finds them.
    summary["trials"][0]["metrics"] = {
        "checkpoint": "/tmp/checkpoint_000007",
    }
    required = _required_tune_checkpoints(summary)
    assert required == {
        "iter_2": "checkpoint_000001",
        "iter_8": "checkpoint_000007",
        "iter_22": "checkpoint_000021",
    }


def test_required_tune_checkpoints_requires_intermediate_names():
    summary = {"trials": [{"checkpoint": "/tmp/checkpoint_000021"}]}
    with pytest.raises(FileNotFoundError, match="checkpoint_000001"):
        _required_tune_checkpoints(summary)


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_probe_leaves_import_and_encode_variant(variant):
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_4."
        f"belief_symmetry_probes_0040.variant_{variant}.experiment"
    )
    assert module.CYCLE == 4
    assert module.VARIANT == variant
    assert callable(module.run)
    assert Path(module.__file__).name == "experiment.py"


def test_campaign_constants():
    assert CAMPAIGN_SUFFIX == "0040"
    assert PROBE_ITERATIONS == (2, 8, 22)
    assert CHECKPOINT_LABELS == ("initial", "iter_2", "iter_8", "iter_22")
