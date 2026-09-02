"""Focused tests for experiment-local batched Cassandra PPO."""

from __future__ import annotations

import torch

from experiments.cassandra_belief_factoring_2026_08.batched_ppo_250m.environment import (
    OBSERVATION_DIM,
    BatchedCassandraEnv,
)
from experiments.cassandra_belief_factoring_2026_08.batched_ppo_250m.global_alias.experiment import (
    ACTION_SCOPE as GLOBAL_SCOPE,
    build_config as build_global,
)
from experiments.cassandra_belief_factoring_2026_08.batched_ppo_250m.model import (
    BatchedTransformerActorCritic,
)
from experiments.cassandra_belief_factoring_2026_08.batched_ppo_250m.shared import (
    CONTEXT_LEN,
    TOTAL_ENV_STEPS,
)
from experiments.cassandra_belief_factoring_2026_08.batched_ppo_250m.targeted.experiment import (
    ACTION_SCOPE as TARGETED_SCOPE,
    build_config as build_targeted,
)
from experiments.cassandra_belief_factoring_2026_08.batched_ppo_250m.trainer import (
    BatchedPPOTrainer,
    TrainerConfig,
    compute_gae,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path, *, resume_from=None):
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        resume_from=resume_from,
        hardware=PROFILES["cpu"],
    )


def test_condition_configs_preserve_science_and_change_only_scope(tmp_path):
    context = _context(tmp_path)
    targeted = build_targeted(context)
    global_alias = build_global(context)

    assert targeted is not build_targeted(context)
    assert targeted.action_scope == TARGETED_SCOPE
    assert global_alias.action_scope == GLOBAL_SCOPE
    assert targeted.total_env_steps == global_alias.total_env_steps == 128
    assert targeted.context_len == global_alias.context_len == CONTEXT_LEN
    assert CONTEXT_LEN == 64
    assert targeted.gamma == global_alias.gamma == 0.990
    assert targeted.gae_lambda == global_alias.gae_lambda == 0.95
    assert targeted.vf_clip_param == global_alias.vf_clip_param == 100.0
    assert targeted.vf_loss_coeff == global_alias.vf_loss_coeff == 0.01
    assert targeted.entropy_coeff == global_alias.entropy_coeff == 0.03
    assert targeted.compile_model is False
    assert TOTAL_ENV_STEPS == 250_000_000


def test_full_cuda_recipe_enables_safe_compilation(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        hardware=PROFILES["cuda4090"],
    )

    config = build_targeted(context)

    assert config.context_len == 64
    assert config.compile_model is True


def test_batched_environment_is_seeded_and_stays_on_device():
    first = BatchedCassandraEnv(
        num_envs=8,
        action_scope="targeted",
        episode_length=3,
        seed=7,
        device=torch.device("cpu"),
    )
    second = BatchedCassandraEnv(
        num_envs=8,
        action_scope="targeted",
        episode_length=3,
        seed=7,
        device=torch.device("cpu"),
    )

    first_obs = first.reset()
    second_obs = second.reset()
    actions = torch.zeros(8, dtype=torch.long)
    first_next, first_rewards, truncated = first.step(actions)
    second_next, second_rewards, _ = second.step(actions)

    assert first_obs.shape == (8, OBSERVATION_DIM)
    assert first_obs.device.type == "cpu"
    assert torch.equal(first_obs, second_obs)
    assert torch.unique(first.components, dim=0).shape[0] > 1
    assert torch.equal(first_next, second_next)
    assert torch.equal(first_rewards, second_rewards)
    assert truncated is False


def test_batched_environment_preserves_action_scope_semantics():
    global_env = BatchedCassandraEnv(
        num_envs=2,
        action_scope="global_aliases",
        episode_length=3,
        seed=7,
        device=torch.device("cpu"),
    )
    global_env.reset()
    global_env.components.zero_()
    global_obs, global_rewards, _ = global_env.step(
        torch.tensor([6, 9])
    )

    assert torch.equal(global_env.components, torch.full((2, 4), 3))
    torch.testing.assert_close(global_rewards, torch.full((2,), -15.0))
    assert torch.equal(
        global_obs[:, :-1].reshape(2, 4, 4).argmax(dim=-1),
        global_env.components,
    )
    torch.testing.assert_close(global_obs[:, -1], global_rewards)

    targeted_env = BatchedCassandraEnv(
        num_envs=2,
        action_scope="targeted",
        episode_length=3,
        seed=7,
        device=torch.device("cpu"),
    )
    targeted_env.reset()
    targeted_env.components.zero_()
    _, targeted_rewards, _ = targeted_env.step(torch.tensor([6, 9]))

    assert torch.equal(
        targeted_env.components,
        torch.tensor([[3, 0, 0, 0], [0, 0, 0, 3]]),
    )
    torch.testing.assert_close(
        targeted_rewards, torch.full((2,), -3.75)
    )


def test_gae_bootstraps_but_does_not_cross_truncation():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.zeros_like(rewards)
    boundaries = torch.tensor([True, False])
    truncated_bootstraps = torch.tensor([[4.0], [0.0]])

    advantages = compute_gae(
        rewards=rewards,
        values=values,
        final_values=torch.zeros(1),
        boundaries=boundaries,
        truncated_bootstraps=truncated_bootstraps,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([[5.0], [2.0]]))


def test_context_ring_materializes_left_padding_and_recent_history():
    model = BatchedTransformerActorCritic(
        observation_dim=2,
        action_count=2,
        d_model=8,
        n_layers=1,
        n_heads=1,
        context_len=3,
    )
    state = model.initial_state(1, torch.device("cpu"))

    with torch.inference_mode():
        for value in (1.0, 2.0):
            _, _, state = model.inference(
                torch.tensor([[value, -value]]), state
            )
    expected_padded = torch.tensor(
        [[[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]]
    )
    torch.testing.assert_close(model.ordered_context(state), expected_padded)

    with torch.inference_mode():
        for value in (3.0, 4.0):
            _, _, state = model.inference(
                torch.tensor([[value, -value]]), state
            )
    expected_recent = torch.tensor(
        [[[2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]]
    )
    torch.testing.assert_close(model.ordered_context(state), expected_recent)


def test_tiny_training_and_checkpoint_resume(tmp_path):
    config = TrainerConfig(
        action_scope="targeted",
        total_env_steps=8,
        num_envs=2,
        rollout_steps=4,
        minibatch_size=4,
        num_epochs=1,
        episode_length=3,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        vf_clip_param=100.0,
        vf_loss_coeff=0.01,
        entropy_coeff=0.03,
        d_model=8,
        n_layers=1,
        n_heads=1,
        context_len=4,
        checkpoint_interval=8,
    )
    context = _context(tmp_path)
    trainer = BatchedPPOTrainer(
        config=config,
        context=context,
        device=torch.device("cpu"),
    )

    summary = trainer.train()
    checkpoint = context.artifacts_dir / "checkpoints/final/checkpoint.pt"
    resumed = BatchedPPOTrainer(
        config=config,
        context=_context(tmp_path, resume_from=checkpoint),
        device=torch.device("cpu"),
    )

    assert summary["status"] == "completed"
    assert summary["env_steps"] == 8
    assert checkpoint.is_file()
    assert resumed.total_env_steps == trainer.total_env_steps
    assert resumed.iteration == trainer.iteration
    for expected, actual in zip(
        trainer.model.parameters(), resumed.model.parameters()
    ):
        torch.testing.assert_close(expected, actual)
