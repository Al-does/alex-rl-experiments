"""Recipe, PPO-mechanics, and smoke tests for the paper-vs-PPO MESS3 study."""

from __future__ import annotations

import json

import pytest
import torch

from experiments.mess3_paper_belief_geometry_ppo_2026_09.mess3 import (
    AliasTable,
    enumerate_paths,
    path_probabilities,
)
from experiments.mess3_paper_belief_geometry_ppo_2026_09.model import (
    PaperActorCritic,
)
from experiments.mess3_paper_belief_geometry_ppo_2026_09.paper_supervised import (
    experiment as paper_experiment,
)
from experiments.mess3_paper_belief_geometry_ppo_2026_09.ppo import (
    PPOConfig,
    collect_rollout,
    ppo_loss,
)
from experiments.mess3_paper_belief_geometry_ppo_2026_09.ppo_token_guess import (
    experiment as ppo_experiment,
)
from experiments.mess3_paper_belief_geometry_ppo_2026_09.shared import (
    PROBE_FIT_SEQUENCES,
    PROBE_TEST_SEQUENCES,
    exact_bayesian_accuracy,
    exact_policy_metrics,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def test_paper_recipe_matches_published_hyperparameters():
    config = paper_experiment.FULL_TRAINING_CONFIG

    assert config.optimizer_name == "sgd"
    assert config.learning_rate == pytest.approx(0.01)
    assert config.batch_size == 64
    assert config.total_steps == 1_000_000
    assert config.weight_decay == 0.0
    assert config.momentum == 0.0
    # Update 3,125 is 3,125 * 64 * 10 = 2M agent steps, the PPO budget.
    assert 3_125 in config.extra_checkpoint_steps
    assert 3_125 * config.batch_size * 10 == PPOConfig().total_agent_steps
    assert not config.retain_periodic_checkpoints
    retained = {
        0,
        config.analyzed_step,
        config.total_steps,
        *config.extra_checkpoint_steps,
    }
    assert len(retained) == 6


def test_probes_use_a_few_thousand_positions_per_split():
    assert PROBE_FIT_SEQUENCES * 10 == 4_000
    assert PROBE_TEST_SEQUENCES * 10 == 4_000


def test_ppo_budget_is_two_million_fresh_agent_steps():
    config = ppo_experiment.FULL_PPO_CONFIG

    assert config.total_agent_steps == 2_000_000
    assert config.iterations(10) == 50
    assert config.iterations(10) // config.checkpoint_every_iterations == 5
    assert config.num_epochs * (
        config.rollout_sequences // config.minibatch_sequences
    ) == 48
    with pytest.raises(ValueError):
        PPOConfig(total_agent_steps=2_000_001).iterations(10)


def test_actor_critic_is_causal_and_keeps_probe_locations():
    torch.manual_seed(0)
    model = PaperActorCritic().eval()
    tokens = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]])
    changed = tokens.clone()
    changed[0, 6:] = 2

    logits, values = model.policy_and_value(tokens)
    changed_logits, changed_values = model.policy_and_value(changed)

    assert logits.shape == (1, 10, 3) and values.shape == (1, 10)
    torch.testing.assert_close(logits[:, :6], changed_logits[:, :6])
    torch.testing.assert_close(values[:, :6], changed_values[:, :6])
    assert model.activation_names[-2:] == ("block_3", "final_ln")


def _rollout(sequences: int = 64):
    paths = enumerate_paths(11)
    probabilities = path_probabilities(paths)
    alias = AliasTable.from_probabilities(probabilities, device="cpu")
    generator = torch.Generator().manual_seed(3)
    torch.manual_seed(0)
    model = PaperActorCritic()
    rollout = collect_rollout(
        model, paths, alias, sequences=sequences, generator=generator
    )
    return model, paths, rollout


def test_rollout_rewards_score_the_next_token_only():
    _, _, rollout = _rollout()
    contexts, targets = rollout["contexts"], rollout["targets"]

    assert contexts.shape == targets.shape == (64, 10)
    torch.testing.assert_close(targets[:, :-1], contexts[:, 1:])
    torch.testing.assert_close(
        rollout["rewards"], (rollout["actions"] == targets).float()
    )


def test_ppo_loss_has_no_policy_gradient_outside_the_clip_region():
    model, _, rollout = _rollout(sequences=8)
    config = PPOConfig(value_loss_coeff=0.0)
    batch = dict(rollout)
    batch["advantages"] = torch.ones_like(batch["advantages"])
    # Pretend the rollout policy was far less likely: ratio >> 1 + clip.
    batch["old_log_probs"] = batch["old_log_probs"] - 5.0

    loss, metrics = ppo_loss(model, batch, config)
    loss.backward()

    assert metrics["clip_fraction"] == pytest.approx(1.0)
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.parameters()
    )


def test_exact_bayesian_predictor_bounds_policy_accuracy():
    paths = enumerate_paths(11)
    probabilities = path_probabilities(paths)
    bayes = exact_bayesian_accuracy(paths, probabilities)
    torch.manual_seed(0)
    metrics = exact_policy_metrics(PaperActorCritic(), paths, probabilities)

    assert bayes == pytest.approx(0.6859, abs=1e-4)
    assert metrics["greedy_accuracy"] <= bayes
    assert metrics["cross_entropy_nats"] > 0.80


@pytest.mark.slow
@pytest.mark.parametrize("module", [paper_experiment, ppo_experiment])
def test_smoke_runs_probe_every_retained_checkpoint(module, tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        seed=42,
        hardware=PROFILES["cpu"],
    )

    summary = module.run(context)

    curve = json.loads(
        (tmp_path / "results" / "checkpoint_probe_curve.json").read_text()
    )
    steps = [point["agent_steps"] for point in curve["checkpoints"]]
    assert steps[0] == 0 and steps == sorted(steps) and len(steps) >= 3
    assert summary["success_checks"]["applicable"] is False
    assert (tmp_path / "results" / "probe_trajectory.png").exists()
