"""Wiring tests for standalone split-PPO reward-stream entropy."""

from __future__ import annotations

import pytest

from experiments.factored_representations_reproduction_PPO_2026_08.process import (
    FACTOR_COUNTS,
)
from experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.model import (
    SplitFactoredReproductionActorCritic,
)
from experiments.factored_representations_reproduction_split_PPO_max_ent.experiment import (
    CONDITION,
    ENTROPY_REWARD_COEFFICIENT,
    PPO_ENTROPY_COEFFICIENT,
    _resolved_recipe,
    build_config,
)
from experiments.mess3_token_guess_cycle_1.entropy_reward import (
    COEFFICIENT_KEY,
    EntropyRewardPPOTorchLearner,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
def test_configs_are_fresh_matched_split_ppo_with_reward_entropy(
    tmp_path,
    factor_count,
):
    context = _context(tmp_path)
    first = build_config(context, factor_count=factor_count)
    second = build_config(context, factor_count=factor_count)

    assert first is not second
    assert first.seed == 42
    assert first.num_env_runners == 0
    assert first.rl_module_spec.module_class is SplitFactoredReproductionActorCritic
    assert first.entropy_coeff == PPO_ENTROPY_COEFFICIENT == 0.0
    assert first.learner_class is EntropyRewardPPOTorchLearner
    assert (
        first.learner_config_dict[COEFFICIENT_KEY]
        == ENTROPY_REWARD_COEFFICIENT
        == 0.5
    )

    recipe = _resolved_recipe(factor_count=factor_count, context=context)
    assert recipe["total_env_steps"] == (1024 if context.smoke else 10_000_000)
    assert recipe["entropy_reward_coefficient"] == 0.5
    assert recipe["ppo_entropy_coefficient"] == 0.0
