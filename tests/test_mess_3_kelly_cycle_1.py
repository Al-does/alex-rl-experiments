"""Focused tests for the Kelly-sized MESS3 token experiment."""

from __future__ import annotations

import math

import numpy as np
import torch

from envs.hmm import HMMEnv
from experiments.mess_3_kelly_cycle_1.kelly import (
    MAX_WAGER,
    expected_log_growth,
    kelly_fraction,
    realized_log_growth,
)
from experiments.mess_3_kelly_cycle_1.learning import (
    FIXED_MODE,
    LEARNED_MODE,
    POLICY_MODE,
    TokenCategoricalWithWager,
    WagerTransformerModel,
    behavior_wager,
)
from experiments.mess_3_kelly_cycle_1.shared import (
    CONDITIONS,
    build_config,
    environment_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models.transformer import TransformerModel


def test_kelly_math_has_fair_three_way_optimum_and_finite_full_loss():
    probabilities = np.array([0.0, 1.0 / 3.0, 0.5, 1.0])
    fractions = kelly_fraction(probabilities)
    np.testing.assert_allclose(
        fractions,
        [0.0, 0.0, 0.25, MAX_WAGER],
    )
    grid = np.linspace(0.0, 0.99, 1_000)
    growth = expected_log_growth(0.6, grid)
    optimum = grid[np.argmax(growth)]
    assert abs(optimum - 0.4) < 0.002
    assert math.isfinite(float(realized_log_growth(False, MAX_WAGER)))


def test_behavior_wagers_cover_fixed_policy_and_learned_modes():
    token_logits = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    actions = torch.tensor([[0, 0]])
    fixed = behavior_wager(
        mode=FIXED_MODE,
        action_dist_inputs=token_logits,
        actions=actions,
    )
    torch.testing.assert_close(fixed, torch.full_like(fixed, MAX_WAGER))

    implied = behavior_wager(
        mode=POLICY_MODE,
        action_dist_inputs=token_logits,
        actions=actions,
    )
    assert implied[0, 0] == 0.0
    assert implied[0, 1] > 0.0

    learned_inputs = torch.cat(
        [token_logits, torch.zeros((*token_logits.shape[:-1], 1))],
        dim=-1,
    )
    learned = behavior_wager(
        mode=LEARNED_MODE,
        action_dist_inputs=learned_inputs,
        actions=actions,
    )
    torch.testing.assert_close(learned, torch.full_like(learned, 0.5))


def test_wager_distribution_ignores_wager_logit_for_token_sampling():
    logits = torch.tensor([[0.0, 5.0, -2.0, 100.0]])
    distribution = TokenCategoricalWithWager.from_logits(logits)
    deterministic = distribution.to_deterministic().sample()
    assert deterministic.tolist() == [1]
    assert distribution.logp(torch.tensor([1])).shape == (1,)


def test_bayes_oracle_uses_internal_belief_without_observation_leakage():
    config = {
        **environment_config("bayes_oracle"),
        "episode_length": 2,
        "diagnostics": {
            "belief": True,
            "tokens": True,
            "rewards": True,
        },
    }
    environment = HMMEnv(config)
    try:
        observation, info = environment.reset(seed=9)
        assert observation.shape == (3,)
        assert environment.task.requires_belief
        guess = info["raw_token_current"]
        probability = float(
            info["belief_current"]
            @ environment.model.emission_matrix[:, guess]
        )
        wager = float(kelly_fraction(probability))
        _, reward, _, _, next_info = environment.step(guess)
        assert reward == float(realized_log_growth(True, wager))
        components = next_info["reward_components"]
        assert components["kelly_wager"] == wager
        assert components["token_guess_correct"] == 1.0
    finally:
        environment.close()


def test_all_conditions_build_fresh_configs_without_aux_or_warm_start(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    configs = {
        condition: build_config(context, condition)
        for condition in CONDITIONS
    }
    for condition, config in configs.items():
        assert config.seed == 42
        assert config.gamma == 1.0
        assert config.num_env_runners == 0
        assert config.train_batch_size_per_learner == 2_048
        assert config.entropy_coeff == 0.0
        assert "next_token_aux" not in config.rl_module_spec.model_config
        assert config.env_config["observation"] == {"action": None}
        if condition == "learned_kelly":
            assert config.rl_module_spec.module_class is WagerTransformerModel
        else:
            assert config.rl_module_spec.module_class is TransformerModel
    assert len({id(config) for config in configs.values()}) == len(CONDITIONS)
