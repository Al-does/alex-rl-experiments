"""Scientific and wiring tests for cycle-2 discrete SAC."""

from __future__ import annotations

import importlib
import math

import pytest
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner

from experiments.factored_representations_reproduction_SAC_2026_08.learning import (
    SACWithNextJointTokenAux,
)
from experiments.factored_representations_reproduction_SAC_cycle_2_2026_08.benchmark_minibatch_capacity import (
    DEFAULT_CANDIDATES,
    MINIBATCH_COUNT,
    largest_completed,
)
from experiments.factored_representations_reproduction_SAC_cycle_2_2026_08.shared import (
    AUXILIARY_COEFFICIENTS,
    LEARNING_STARTS,
    PRIORITIZED_REPLAY_ALPHA,
    PRIORITIZED_REPLAY_BETA,
    TARGET_ENTROPY_FRACTIONS,
    TRAIN_BATCH_SIZE,
    TRAINING_INTENSITY,
    build_config,
    target_entropy,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from losses.next_token import LAMBDA_KEY


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090_gpuinfer"],
    )


@pytest.mark.parametrize("factor_count", [2, 3])
@pytest.mark.parametrize("entropy_fraction", TARGET_ENTROPY_FRACTIONS)
def test_reward_only_cells_resolve_preregistered_cycle_2_values(
    tmp_path,
    factor_count,
    entropy_fraction,
):
    config = build_config(
        _context(tmp_path),
        factor_count=factor_count,
        condition="sac",
        target_entropy_fraction=entropy_fraction,
    )

    assert config.train_batch_size_per_learner == TRAIN_BATCH_SIZE == 256
    assert config.num_steps_sampled_before_learning_starts == LEARNING_STARTS == 10_000
    assert config.training_intensity == TRAINING_INTENSITY == 1.0
    assert config.replay_buffer_config["alpha"] == PRIORITIZED_REPLAY_ALPHA == 0.6
    assert config.replay_buffer_config["beta"] == PRIORITIZED_REPLAY_BETA == 0.6
    assert config.target_entropy == pytest.approx(
        entropy_fraction * math.log(3**factor_count)
    )
    assert config.learner_class is SACTorchLearner
    assert config.rollout_fragment_length == 1
    assert config.torch_compile_learner is True


@pytest.mark.parametrize("factor_count", [2, 3])
@pytest.mark.parametrize("entropy_fraction", TARGET_ENTROPY_FRACTIONS)
@pytest.mark.parametrize("auxiliary_coefficient", AUXILIARY_COEFFICIENTS)
def test_auxiliary_cells_resolve_two_by_two_sweep(
    tmp_path,
    factor_count,
    entropy_fraction,
    auxiliary_coefficient,
):
    config = build_config(
        _context(tmp_path),
        factor_count=factor_count,
        condition="sac_aux_ce",
        target_entropy_fraction=entropy_fraction,
        auxiliary_coefficient=auxiliary_coefficient,
    )

    assert config.learner_class is SACWithNextJointTokenAux
    assert config.learner_config_dict[LAMBDA_KEY] == auxiliary_coefficient
    assert config.rl_module_spec.model_config["next_token_aux"]["num_classes"] == (
        3**factor_count
    )
    assert config.target_entropy == pytest.approx(
        entropy_fraction * math.log(3**factor_count)
    )


@pytest.mark.parametrize("factor_count", [2, 3])
def test_entropy_targets_are_positive_fractions_of_categorical_maximum(factor_count):
    maximum = math.log(3**factor_count)
    values = [
        target_entropy(factor_count, fraction)
        for fraction in TARGET_ENTROPY_FRACTIONS
    ]

    assert values == sorted(values)
    assert all(0.0 < value < maximum for value in values)


def test_reward_only_rejects_auxiliary_coefficient(tmp_path):
    with pytest.raises(ValueError, match="does not accept"):
        build_config(
            _context(tmp_path),
            factor_count=2,
            condition="sac",
            target_entropy_fraction=0.3,
            auxiliary_coefficient=0.1,
        )


@pytest.mark.parametrize(
    "arm",
    [
        "sac_entropy_0p3",
        "sac_entropy_0p6",
        "sac_aux_0p1_entropy_0p3",
        "sac_aux_0p1_entropy_0p6",
        "sac_aux_0p3_entropy_0p3",
        "sac_aux_0p3_entropy_0p6",
    ],
)
def test_each_preregistered_arm_exports_run(arm):
    module = importlib.import_module(
        "experiments.factored_representations_reproduction_SAC_cycle_2_2026_08."
        f"{arm}.experiment"
    )
    assert callable(module.run)


def test_capacity_candidates_split_into_exactly_eight_minibatches():
    assert MINIBATCH_COUNT == 8
    assert all(batch_size % MINIBATCH_COUNT == 0 for batch_size in DEFAULT_CANDIDATES)
    assert [batch_size // MINIBATCH_COUNT for batch_size in DEFAULT_CANDIDATES] == [
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
    ]


def test_largest_completed_ignores_failed_larger_candidate():
    results = [
        {"batch_size": 65_536, "status": "completed"},
        {"batch_size": 131_072, "status": "completed"},
        {"batch_size": 262_144, "status": "oom"},
    ]
    assert largest_completed(results) == results[1]
