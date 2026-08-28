"""Scientific and wiring tests for the discrete-SAC reproduction."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_SAC_2026_08.learning import (
    AUXILIARY_COEFFICIENT,
    SACWithNextJointTokenAux,
    next_joint_token_targets,
)
from experiments.factored_representations_reproduction_SAC_2026_08.model import (
    FactoredReproductionSAC,
    ReproductionSACEncoder,
)
from experiments.factored_representations_reproduction_SAC_2026_08.process import (
    CONTEXT_LENGTH,
    FACTOR_COUNTS,
    environment_config,
    joint_token_count,
)
from experiments.factored_representations_reproduction_SAC_2026_08.shared import (
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_LEARNING_STARTS,
    build_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from losses.next_token import FWD_KEY


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def _module(*, token_count: int = 9, auxiliary: bool = False):
    model_config = {
        **MODEL_CONFIG,
        "twin_q": True,
    }
    if auxiliary:
        model_config["next_token_aux"] = {"num_classes": token_count}
    module = FactoredReproductionSAC(
        observation_space=gym.spaces.Box(
            0.0,
            1.0,
            shape=(CONTEXT_LENGTH * token_count,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(token_count),
        model_config=model_config,
        inference_only=False,
    )
    module.make_target_networks()
    return module


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
def test_history_observation_preserves_bos_and_revealed_tokens(factor_count):
    token_count = joint_token_count(factor_count)
    environment = HMMEnv(environment_config(factor_count))
    try:
        observation, info = environment.reset(seed=7)
        assert observation.shape == (CONTEXT_LENGTH * token_count,)
        assert observation.sum() == 0.0

        hidden_joint_token = info["raw_token_current"]
        next_observation, reward, _, _, next_info = environment.step(
            hidden_joint_token
        )
        assert reward == 1.0
        assert next_info["visible_source_token"] == hidden_joint_token
        history = next_observation.reshape(CONTEXT_LENGTH, token_count)
        assert history[0].argmax() == hidden_joint_token
        assert history[0].sum() == 1.0
        assert history[1:].sum() == 0.0
    finally:
        environment.close()


def test_actor_and_critics_are_entirely_separate_transformers_with_linear_heads():
    module = _module()
    assert isinstance(module.pi_encoder, ReproductionSACEncoder)
    assert isinstance(module.qf_encoder, ReproductionSACEncoder)
    assert isinstance(module.qf_twin_encoder, ReproductionSACEncoder)
    assert module.pi_encoder is not module.qf_encoder
    assert module.pi_encoder is not module.qf_twin_encoder
    assert module.qf_encoder is not module.qf_twin_encoder

    actor_parameters = {id(value) for value in module.pi_encoder.parameters()}
    critic_parameters = {id(value) for value in module.qf_encoder.parameters()}
    twin_parameters = {id(value) for value in module.qf_twin_encoder.parameters()}
    assert actor_parameters.isdisjoint(critic_parameters)
    assert actor_parameters.isdisjoint(twin_parameters)
    assert critic_parameters.isdisjoint(twin_parameters)

    assert module.reproduction_config.d_model == 64
    assert module.reproduction_config.n_layers == 4
    assert module.reproduction_config.n_heads == 4
    assert module.encoder.token_embedding_matrix().shape == (9, 64)
    assert len(module.pi.net.mlp) == 1
    assert len(module.qf.net.mlp) == 1


def test_actor_hidden_probe_is_pre_final_norm_and_drives_policy_only():
    torch.manual_seed(3)
    module = _module().eval()
    observations = torch.zeros((2, CONTEXT_LENGTH * 9))
    observations[1, 0] = 1.0

    actor_hidden = module.pi_encoder.encode_pre_final_norm(observations)
    normalized = module.encoder.final_norm(actor_hidden)
    inference = module.forward_inference({Columns.OBS: observations})
    torch.testing.assert_close(
        inference[Columns.ACTION_DIST_INPUTS],
        module.pi(normalized),
    )
    assert actor_hidden.shape == (2, 64)
    assert not torch.allclose(actor_hidden, normalized)


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
@pytest.mark.parametrize("condition", ["sac", "sac_aux_ce"])
def test_smoke_configs_are_fresh_and_resolve_each_sac_design_cell(
    tmp_path,
    factor_count,
    condition,
):
    context = _context(tmp_path)
    first = build_config(
        context,
        factor_count=factor_count,
        condition=condition,
    )
    second = build_config(
        context,
        factor_count=factor_count,
        condition=condition,
    )

    assert first is not second
    assert first.seed == 42
    assert first.gamma == 0.0
    assert first.n_step == 1
    assert first.twin_q is True
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.num_steps_sampled_before_learning_starts == SMOKE_LEARNING_STARTS
    assert first.num_env_runners == 0
    assert first.rl_module_spec.module_class is FactoredReproductionSAC
    assert first.rl_module_spec.model_config["d_model"] == 64

    environment = first.env(first.env_config)
    try:
        assert environment.action_space.n == 3**factor_count
        assert environment.observation_space.shape == (
            CONTEXT_LENGTH * 3**factor_count,
        )
        assert environment.config.delay == 1
    finally:
        environment.close()

    if condition == "sac_aux_ce":
        assert first.learner_class is SACWithNextJointTokenAux
        assert (
            first.learner_config_dict["next_token_aux/lambda"]
            == AUXILIARY_COEFFICIENT
        )
        assert (
            first.rl_module_spec.model_config["next_token_aux"]["num_classes"]
            == 3**factor_count
        )
    else:
        assert first.learner_class is SACTorchLearner


def test_auxiliary_head_uses_actor_embedding_and_next_history_token():
    module = _module(auxiliary=True)
    observations = torch.zeros((4, CONTEXT_LENGTH * 9))
    next_observations = observations.clone()
    next_observations[torch.arange(4), torch.tensor([1, 3, 5, 7])] = 1.0
    outputs = module.forward_train(
        {
            Columns.OBS: observations,
            Columns.NEXT_OBS: next_observations,
        }
    )
    logits, targets, valid = next_joint_token_targets(
        {Columns.NEXT_OBS: next_observations},
        outputs[FWD_KEY],
    )

    assert logits.shape == (4, 9)
    torch.testing.assert_close(targets, torch.tensor([1, 3, 5, 7]))
    assert valid.all()
    assert hasattr(module.pi_encoder, "next_token_aux_head")
    assert not hasattr(module.qf_encoder, "next_token_aux_head")
    assert not hasattr(module.qf_twin_encoder, "next_token_aux_head")
