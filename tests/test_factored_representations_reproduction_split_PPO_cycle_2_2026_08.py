"""Wiring tests for cycle-2 PPO with split actor and critic networks."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from experiments.factored_representations_reproduction_PPO_2026_08.process import (
    FACTOR_COUNTS,
)
from experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.model import (
    SplitActorCriticWithNextJointTokenAux,
    SplitFactoredReproductionActorCritic,
)
from experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.shared import (
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_MINIBATCH_SIZE,
    build_config,
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


def _module() -> SplitFactoredReproductionActorCritic:
    return SplitFactoredReproductionActorCritic(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(9,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(9),
        model_config=MODEL_CONFIG,
    )


def test_actor_and_critic_transformers_have_disjoint_parameters():
    module = _module()

    actor_parameters = {id(parameter) for parameter in module.actor_encoder.parameters()}
    critic_parameters = {
        id(parameter) for parameter in module.critic_encoder.parameters()
    }

    assert actor_parameters
    assert critic_parameters
    assert actor_parameters.isdisjoint(critic_parameters)
    assert module.encoder is module.actor_encoder
    assert module.heads.policy.weight is not module.heads.value.weight


def test_policy_and_probe_use_actor_while_values_use_critic():
    torch.manual_seed(7)
    module = _module()
    observations = torch.zeros((2, 3, 9))
    observations[:, 1:, 4] = 1.0
    initial = module.get_initial_state()
    state = {
        key: torch.from_numpy(value).unsqueeze(0).repeat(2, *([1] * value.ndim))
        for key, value in initial.items()
    }
    batch = {
        Columns.OBS: observations,
        Columns.STATE_IN: state,
    }

    actor_embeddings, _ = module._encode_train(batch)
    policy_logits = module.action_distribution_inputs(actor_embeddings)
    values = module.compute_values(batch, embeddings=actor_embeddings)
    probe_residuals = module.encode_chunks_pre_final_norm(
        state["ctx"],
        state["len"].reshape(-1),
        observations,
    )
    direct_actor_residuals = module.actor_encoder(
        state["ctx"],
        state["len"].reshape(-1),
        observations,
        apply_final_norm=False,
    )

    assert policy_logits.shape == (2, 3, 9)
    assert values.shape == (2, 3)
    torch.testing.assert_close(probe_residuals, direct_actor_residuals)

    policy_logits.sum().backward()
    assert any(parameter.grad is not None for parameter in module.actor_encoder.parameters())
    assert all(parameter.grad is None for parameter in module.critic_encoder.parameters())


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
@pytest.mark.parametrize("condition", ["ppo", "ppo_aux_ce"])
def test_smoke_configs_are_fresh_and_select_split_modules(
    tmp_path,
    factor_count,
    condition,
):
    context = _context(tmp_path)
    first = build_config(context, factor_count=factor_count, condition=condition)
    second = build_config(context, factor_count=factor_count, condition=condition)

    assert first is not second
    assert first.seed == 42
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert first.num_env_runners == 0
    assert first.rl_module_spec.module_class is (
        SplitActorCriticWithNextJointTokenAux
        if condition == "ppo_aux_ce"
        else SplitFactoredReproductionActorCritic
    )
