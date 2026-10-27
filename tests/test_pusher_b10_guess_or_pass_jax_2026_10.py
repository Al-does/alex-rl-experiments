from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from envs.hmm import ActionDecision, TransitionEvent
from harness.cli import execute_experiment, load_experiment
from harness.context import RunContext
from harness.hardware import PROFILES

from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
    ReproductionResidualEncoder,
)
from experiments.pusher_b.rl import MODEL_CONFIG
from experiments.pusher_b10_guess_or_pass_2026_10.campaign import (
    campaign_runs as rl_campaign_runs,
)
from experiments.pusher_b10_guess_or_pass_2026_10.task import GuessOrPassTask
from experiments.pusher_b10_guess_or_pass_jax_2026_10 import campaign
from experiments.pusher_b10_guess_or_pass_jax_2026_10.convert import (
    torch_encoder_to_jax,
)
from experiments.pusher_b10_guess_or_pass_jax_2026_10.ppo import (
    LR_SCHEDULE,
    PPOConfig,
    bayes_expected_reward,
    init,
    make_update,
)
from experiments.pusher_b10_guess_or_pass_jax_2026_10.process import (
    guess_or_pass_reward,
    make_env,
)
from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.process import BOS_TOKEN, pusher_b_model


def _torch_state_with_encoder_prefix(encoder: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        f"encoder.{key}": value.detach().cpu().numpy().copy()
        for key, value in encoder.state_dict().items()
    }


def _assert_tree_equal(left, right):
    left_leaves, left_def = jax.tree.flatten(left)
    right_leaves, right_def = jax.tree.flatten(right)
    assert left_def == right_def
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_tree_changed(left, right):
    left_leaves, left_def = jax.tree.flatten(left)
    right_leaves, right_def = jax.tree.flatten(right)
    assert left_def == right_def
    assert any(
        not np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def test_guess_or_pass_reward_correct_wrong_and_pass():
    reward = guess_or_pass_reward(1.5)
    actions = jnp.asarray([0, 1, 2, 0, 1, 2])
    tokens = jnp.asarray([0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(
        np.asarray(reward(actions, tokens)),
        np.asarray([1.0, -1.5, 0.0, -1.5, 1.0, 0.0]),
    )


def test_torch_encoder_conversion_matches_complete_episode_forward():
    torch.manual_seed(17)
    model_config = FactoredReproductionModelConfig.from_dict(MODEL_CONFIG)
    encoder = ReproductionResidualEncoder(2, model_config).eval()
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.copy_(0.1 * torch.randn_like(parameter))
    state = _torch_state_with_encoder_prefix(encoder)
    spec = tm.ModelSpec()
    converted = torch_encoder_to_jax(state, spec)

    rng = np.random.default_rng(21)
    tokens = rng.integers(0, 2, size=(3, 127), dtype=np.int32)
    observations = np.zeros((3, 127, 2), dtype=np.float32)
    observations[:, 1:] = np.eye(2, dtype=np.float32)[tokens[:, :-1]]
    policy_tokens = np.concatenate(
        [np.full((3, 1), BOS_TOKEN, dtype=np.int32), tokens[:, :-1]], axis=1
    )
    with torch.no_grad():
        expected = encoder.forward_complete_episode(torch.from_numpy(observations))
    actual = tm.encode(
        spec, converted, jnp.asarray(policy_tokens, dtype=jnp.int32)
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), atol=1e-5, rtol=0.0
    )


def test_torch_encoder_conversion_rejects_extra_and_missing_encoder_keys():
    torch.manual_seed(7)
    encoder = ReproductionResidualEncoder(
        2, FactoredReproductionModelConfig.from_dict(MODEL_CONFIG)
    )
    state = _torch_state_with_encoder_prefix(encoder)
    with pytest.raises(ValueError, match="unconsumed source encoder keys"):
        torch_encoder_to_jax(
            {
                **state,
                "encoder.position_embedding.weight": np.zeros((128, 128)),
            },
            tm.ModelSpec(),
        )
    missing = dict(state)
    del missing["encoder.blocks.0.attention.qkv.weight"]
    with pytest.raises(ValueError, match="missing keys"):
        torch_encoder_to_jax(missing, tm.ModelSpec())


def test_bayes_expected_reward_matches_exact_numpy_task_filter():
    model = pusher_b_model("b10")
    env = make_env(1.5)
    tokens = np.asarray(
        env.sample_tokens(jax.random.key(230), batch_size=4, length=127)
    )
    expected = []
    for sequence in tokens:
        task = GuessOrPassTask(model=model, cost=1.5)
        task.reset()
        for step, token in enumerate(sequence):
            decision = ActionDecision(
                requested_action=2,
                executed_action=2,
                transition_matrix=model.transition_matrix,
            )
            event = TransitionEvent(
                step=step,
                state_before=0,
                state_after=0,
                raw_token_before=int(token),
                raw_token_after=int(token),
            )
            _, metrics = task.reward(event, decision)
            expected.append(metrics["bayes_expected_reward"])
    actual = bayes_expected_reward(jnp.asarray(tokens), env, 1.5)
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1), np.asarray(expected), atol=1e-6, rtol=0
    )


def test_frozen_and_full_updates_change_only_the_expected_parameter_subtrees():
    config = PPOConfig.smoke(lr_schedule=LR_SCHEDULE)
    spec = tm.ModelSpec()
    env = make_env(2.0)

    frozen = init(config, spec, jax.random.key(9), "frozen")
    trunk_before = {
        key: frozen.params[key]
        for key in ("embed", "blocks", "final_norm")
    }
    frozen_update = jax.jit(make_update(config, env, spec, 2.0, "frozen"))
    frozen_metrics = None
    for _ in range(2):
        frozen, frozen_metrics = frozen_update(frozen)
    assert bool(frozen_metrics["trunk_bit_identical"])
    assert bool(frozen_metrics["heads_changed"])
    _assert_tree_equal(
        trunk_before,
        {key: frozen.params[key] for key in ("embed", "blocks", "final_norm")},
    )
    jax.clear_caches()

    full = init(config, spec, jax.random.key(10), "full")
    trunk_before = {
        key: full.params[key] for key in ("embed", "blocks", "final_norm")
    }
    full_update = jax.jit(make_update(config, env, spec, 2.0, "full"))
    full, _ = full_update(full)
    _assert_tree_changed(
        trunk_before,
        {key: full.params[key] for key in ("embed", "blocks", "final_norm")},
    )
    jax.clear_caches()


def test_campaign_order_commands_and_all_indices(monkeypatch, capsys):
    for frozen_cost in (2.0, 1.5, 1.0):
        actual = campaign.campaign_runs(frozen_cost)
        expected = rl_campaign_runs(frozen_cost)
        assert len(actual) == len(expected) == 27
        assert [
            (run.leaf, run.source, run.trunk, run.cost, run.seed)
            for run in actual
        ] == [
            (run.leaf, run.source, run.trunk, run.cost, run.seed)
            for run in expected
        ]
        assert all(run.run_name.startswith("guess-or-pass-jax-") for run in actual)
        assert actual[3].source == expected[3].source
        assert actual[3].trunk == "full"
        assert actual[21].trunk == "frozen"
    run = campaign.campaign_runs()[21]
    command = campaign._command(
        run, hardware="cpu", upload_artifacts=False, smoke=True
    )
    assert command[0] == campaign.sys.executable
    assert campaign.TRANSFER_MODULE in command
    assert "pusher_b10_guess_or_pass_jax_2026_10" in " ".join(command)

    invoked = []
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda command, **kwargs: (
            invoked.append(command) or SimpleNamespace(returncode=0)
        ),
    )
    assert campaign.main(["--all", "--frozen-cost", "1.5"]) == 0
    assert len(invoked) == 27
    assert all(command[0] == campaign.sys.executable for command in invoked)
    assert "Campaign exit codes:" in capsys.readouterr().out


def test_scratch_leaf_smoke_writes_summary_and_parameter_artifacts(tmp_path):
    experiment = load_experiment(
        "experiments.pusher_b10_guess_or_pass_jax_2026_10.ppo_c2.experiment"
    )
    context = RunContext(
        experiment_dir=experiment.directory,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        run_id="test-smoke",
        smoke=True,
        hardware=PROFILES["cpu"],
        upload_artifacts=False,
    )
    summary = execute_experiment(
        experiment, context, upload_artifacts=False
    )
    required = {
        "study",
        "algorithm",
        "cost",
        "seed",
        "smoke",
        "target_env_steps",
        "completed_env_steps",
        "episode_return_mean",
        "guess_correct",
        "guess_wrong",
        "passed",
        "bayes_expected_reward",
        "transfer",
        "reward_per_step",
        "compile_seconds",
        "env_steps_per_second_active",
        "device",
        "parameter_count",
    }
    assert required <= summary.keys()
    assert summary["cost"] == 2.0
    assert summary["smoke"] is True
    assert json.loads((tmp_path / "results" / "summary.json").read_text()) == summary
    assert (tmp_path / "results" / "training_curves.jsonl").is_file()
    assert (tmp_path / "artifacts" / "initial_params.npz").is_file()
    assert (tmp_path / "artifacts" / "final_params.npz").is_file()
