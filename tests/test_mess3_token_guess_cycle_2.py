"""Scientific and wiring tests for token-guess cycle 2."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_2.analysis import (
    BAYESIAN_OPTIMAL_ACCURACY,
    FULL_TEST_STEPS,
    PLOT_SAMPLE_SIZE,
    _episode_clusters,
    _permutation_null_metrics,
    bayesian_optimal_accuracy,
)
from experiments.mess3_belief_geometry_2026_07.probe import ProbeData
from experiments.mess3_token_guess_cycle_2.learning import (
    KELLY_LOSS_COEFFICIENT_KEY,
    A2CTorchLearner,
    KellyPPOTorchLearner,
    a2c_objective,
    realized_log_growth,
)
from experiments.mess3_token_guess_cycle_2.model import (
    PaperActorCriticConfig,
    PaperResidualEncoder,
)
from experiments.mess3_token_guess_cycle_2.shared import (
    BASE_MODEL_CONFIG,
    CONDITIONS,
    ENV_CONFIG,
    IQN_CONFIG,
    IQNModel,
    KellyModel,
    PredictiveLearner,
    PredictiveModel,
    VALIDATION_ENV_STEPS,
    build_config,
    checkpoint_records,
    next_emission_targets,
)
from experiments.mess3_token_guess_cycle_2.sweeps import (
    PREDICTIVE_LOSS_COEFFICIENT_KEY,
    SWEEP_SPECS,
    build_sweep_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners import IQNPPOTorchLearner


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_paper_actor_critic_config_matches_requested_architecture():
    assert BASE_MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 1,
        "d_head": 8,
        "d_mlp": 256,
        "context_length": 10,
        "max_seq_len": 32,
        "activation": "relu",
        "normalization": "layer_norm",
        "positional_embedding": "learned_absolute",
    }
    with pytest.raises(ValueError, match="exactly one head"):
        PaperActorCriticConfig(n_heads=2)


def test_vectorized_encoder_is_causal_and_matches_stepwise_windows():
    torch.manual_seed(3)
    config = PaperActorCriticConfig()
    encoder = PaperResidualEncoder(3, config).eval()
    observations = torch.nn.functional.one_hot(
        torch.arange(17).remainder(3),
        num_classes=3,
    ).to(dtype=torch.float32)[None, :, :]
    context = torch.zeros((1, 9, 3))
    lengths = torch.zeros(1)

    chunked = encoder(context, lengths, observations)
    stepwise = []
    step_context = context
    step_lengths = lengths
    for index in range(observations.shape[1]):
        current = observations[:, index : index + 1]
        stepwise.append(encoder(step_context, step_lengths, current))
        step_context = torch.cat([step_context, current], dim=1)[:, -9:, :]
        step_lengths = (step_lengths + 1).clamp(max=9)
    torch.testing.assert_close(
        chunked,
        torch.cat(stepwise, dim=1),
        atol=1e-6,
        rtol=1e-5,
    )

    changed = observations.clone()
    changed[:, 11:, :] = changed[:, 11:, :].roll(1, dims=-1)
    changed_outputs = encoder(context, lengths, changed)
    torch.testing.assert_close(
        chunked[:, :11],
        changed_outputs[:, :11],
        atol=1e-7,
        rtol=1e-6,
    )


def test_next_emission_target_is_the_token_scored_by_delay_one_task():
    observations = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ]
    )
    logits = torch.zeros((1, 4, 3))
    aligned, targets, valid = next_emission_targets(
        {
            Columns.OBS: observations,
            Columns.LOSS_MASK: torch.ones((1, 4), dtype=torch.bool),
        },
        logits,
    )
    assert aligned.shape == (1, 3, 3)
    torch.testing.assert_close(targets, torch.tensor([[1, 2, 0]]))
    assert valid.all()

    environment = HMMEnv(
        {
            **ENV_CONFIG,
            "episode_length": 2,
            "diagnostics": {"tokens": True, "transitions": True},
        }
    )
    try:
        _, info = environment.reset(seed=5)
        expected = info["raw_token_current"]
        next_observation, reward, _, _, step_info = environment.step(expected)
        assert reward == 1.0
        assert step_info["raw_token_before"] == expected
        assert next_observation[expected] == 1.0
    finally:
        environment.close()


def test_bayesian_optimum_is_exact_finite_context_ceiling():
    assert bayesian_optimal_accuracy(context_length=0) == pytest.approx(1 / 3)
    assert bayesian_optimal_accuracy(context_length=1) == pytest.approx(
        0.6736875
    )
    assert BAYESIAN_OPTIMAL_ACCURACY == pytest.approx(
        0.6895773959227975,
        abs=1e-15,
    )
    assert FULL_TEST_STEPS == 80_000
    assert PLOT_SAMPLE_SIZE == 80_000


def _probe_data(
    activations: np.ndarray,
    beliefs: np.ndarray,
    *,
    env_indices: np.ndarray | None = None,
    episode_steps: np.ndarray | None = None,
) -> ProbeData:
    count = len(beliefs)
    zeros = np.zeros(count, dtype=np.int64)
    return ProbeData(
        activations=activations,
        beliefs=beliefs,
        diagnostic_beliefs=beliefs,
        tokens=zeros,
        previous_tokens=zeros,
        env_indices=zeros if env_indices is None else env_indices,
        episode_steps=(
            np.arange(count, dtype=np.int64)
            if episode_steps is None
            else episode_steps
        ),
        states=zeros,
        actions=zeros.astype(np.float64),
        rewards=zeros.astype(np.float64),
    )


def test_episode_clusters_preserve_environment_episode_dependence():
    data = _probe_data(
        np.zeros((8, 2)),
        np.zeros((8, 3)),
        env_indices=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        episode_steps=np.array([4, 4, 5, 5, 0, 6, 1, 0]),
    )
    clusters = _episode_clusters(data)
    assert clusters[0] == clusters[2]
    assert clusters[4] == clusters[6]
    assert clusters[0] != clusters[4]
    assert clusters[1] == clusters[3] == clusters[5]
    assert clusters[1] != clusters[7]


def test_permutation_null_reports_readme_metrics():
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(120, 4))
    test_x = rng.normal(size=(80, 4))
    weight = rng.normal(size=(4, 3))
    train_y = train_x @ weight
    test_y = test_x @ weight
    metrics = _permutation_null_metrics(
        _probe_data(train_x, train_y),
        _probe_data(test_x, test_y),
        n_permutations=20,
        sample_seed=11,
        permutation_seed=12,
    )
    assert metrics["permutation_real_mse"] < 1e-10
    assert metrics["permutation_null_mse_p05"] > 0.1
    assert metrics["permutation_null_n"] == 20
    assert metrics["permutation_null_p_value_lower_tail"] == pytest.approx(
        1 / 21
    )


def test_a2c_objective_matches_masked_manual_calculation_and_backpropagates():
    logp = torch.tensor([[0.2, -0.3, 4.0]], requires_grad=True)
    advantages = torch.tensor([[2.0, -1.0, 99.0]])
    values = torch.tensor([[0.5, 0.0, 8.0]], requires_grad=True)
    targets = torch.tensor([[1.0, -1.0, -9.0]])
    entropy = torch.tensor([[0.7, 0.5, 3.0]])
    mask = torch.tensor([[True, True, False]])
    total, policy, value, mean_entropy = a2c_objective(
        logp=logp,
        advantages=advantages,
        values=values,
        value_targets=targets,
        entropy=entropy,
        loss_mask=mask,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
    )
    assert policy.item() == pytest.approx(-0.35)
    assert value.item() == pytest.approx(0.3125)
    assert mean_entropy.item() == pytest.approx(0.6)
    assert total.item() == pytest.approx(-0.35 + 0.5 * 0.3125 - 0.01 * 0.6)
    total.backward()
    assert logp.grad is not None
    assert values.grad is not None
    assert logp.grad[0, 2] == 0.0
    assert values.grad[0, 2] == 0.0


def test_direct_kelly_math_stays_differentiable_and_action_conditional():
    wagers = torch.tensor([0.5, 0.25], requires_grad=True)
    growth = realized_log_growth(
        torch.tensor([True, False]),
        wagers,
    )
    torch.testing.assert_close(
        growth,
        torch.tensor([np.log(2.0), np.log(0.75)], dtype=wagers.dtype),
    )
    (-growth.mean()).backward()
    assert wagers.grad is not None
    assert torch.count_nonzero(wagers.grad) == 2


def test_five_conditions_build_fresh_gamma_zero_configs(tmp_path):
    context = _context(tmp_path)
    configs = {
        condition.name: build_config(context, condition.name)
        for condition in CONDITIONS
    }
    assert set(configs) == {
        "a2c",
        "ppo",
        "predictive_loss",
        "decoupled_kelly",
        "iqn",
    }
    for config in configs.values():
        assert config.gamma == 0.0
        assert config.lambda_ == 0.0
        assert config.train_batch_size_per_learner == 2_048
        assert config.env_config == ENV_CONFIG
        assert config.num_env_runners == 0
    assert configs["a2c"].learner_class is A2CTorchLearner
    assert configs["a2c"].num_epochs == 1
    assert configs["a2c"].minibatch_size is None
    assert configs["ppo"].learner_class is PPOTorchLearner
    assert configs["predictive_loss"].learner_class is PredictiveLearner
    assert configs["predictive_loss"].rl_module_spec.module_class is PredictiveModel
    assert configs["decoupled_kelly"].learner_class is KellyPPOTorchLearner
    assert configs["decoupled_kelly"].rl_module_spec.module_class is KellyModel
    assert (
        configs["decoupled_kelly"].learner_config_dict[
            KELLY_LOSS_COEFFICIENT_KEY
        ]
        == 1.0
    )
    assert configs["iqn"].learner_class is IQNPPOTorchLearner
    assert configs["iqn"].rl_module_spec.module_class is IQNModel
    assert configs["iqn"].rl_module_spec.model_config["iqn_value"] == IQN_CONFIG
    assert VALIDATION_ENV_STEPS == 131_072


def test_single_gpu_profile_reserves_cuda_for_learner(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090_gpuinfer"],
    )
    config = build_config(context, "ppo")
    assert config.num_gpus_per_learner == 1
    assert config.num_gpus_per_env_runner == 0


def test_hyperparameter_sweeps_are_four_point_rllib_grids(tmp_path):
    context = _context(tmp_path)
    assert set(SWEEP_SPECS) == {
        "learning_rate",
        "predictive_loss_coefficient",
        "kelly_loss_coefficient",
    }
    assert len(SWEEP_SPECS["learning_rate"].values) == 4
    assert (
        max(SWEEP_SPECS["learning_rate"].values)
        / min(SWEEP_SPECS["learning_rate"].values)
        >= 10
    )
    for spec in SWEEP_SPECS.values():
        assert len(spec.values) == 4
        assert tuple(sorted(spec.values)) == spec.values

    lr_config = build_sweep_config(context, "learning_rate").to_dict()
    predictive_config = build_sweep_config(
        context,
        "predictive_loss_coefficient",
    ).to_dict()
    kelly_config = build_sweep_config(
        context,
        "kelly_loss_coefficient",
    ).to_dict()
    assert lr_config["lr"] == {
        "grid_search": list(SWEEP_SPECS["learning_rate"].values)
    }
    assert predictive_config["learner_config_dict"][
        PREDICTIVE_LOSS_COEFFICIENT_KEY
    ] == {
        "grid_search": list(
            SWEEP_SPECS["predictive_loss_coefficient"].values
        )
    }
    assert kelly_config["learner_config_dict"][
        KELLY_LOSS_COEFFICIENT_KEY
    ] == {
        "grid_search": list(SWEEP_SPECS["kelly_loss_coefficient"].values)
    }


def test_checkpoint_records_include_every_unique_retained_checkpoint():
    result = SimpleNamespace(
        checkpoint=SimpleNamespace(path="/tmp/checkpoint_000002"),
        metrics={
            "training_iteration": 2,
            "env_runners": {"num_env_steps_sampled_lifetime": 4096},
        },
        best_checkpoints=[
            (
                SimpleNamespace(path="/tmp/checkpoint_000002"),
                {
                    "training_iteration": 2,
                    "env_runners": {"num_env_steps_sampled_lifetime": 4096},
                },
            ),
            (
                SimpleNamespace(path="/tmp/checkpoint_000001"),
                {
                    "training_iteration": 1,
                    "env_runners": {"num_env_steps_sampled_lifetime": 2048},
                },
            ),
        ],
    )
    records = checkpoint_records(result)
    assert [record["agent_steps"] for record in records] == [2048, 4096]
