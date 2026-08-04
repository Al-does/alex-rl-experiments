from __future__ import annotations

import json

import numpy as np
import torch

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_asymmetry_cycle_5.task import (
    ActionSymmetryTask as LegacyActionSymmetryTask,
)
from experiments.mess3_belief_geometry_2026_07.probe import ProbeData
from experiments.mess3_reward_state_action_symmetry_cycle_5.compare_next_token_probe import (
    load_joined_rows,
    summarize,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.task import (
    ActionSymmetryTask,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.next_token_probe.probe import (
    NextTokenTransformerProbe,
    ProbeTrainingConfig,
    SequenceDataset,
    build_sequence_dataset,
    exact_next_token_probabilities,
    fit_probe,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.next_token_probe import (
    experiment as probe_experiment,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.shared import (
    BASE_MODEL_CONFIG,
    environment_config,
)
from learners.models.transformer import TransformerModel


def _probe_data() -> ProbeData:
    n = 10
    beliefs = np.tile(np.array([[0.2, 0.3, 0.5]]), (n, 1))
    return ProbeData(
        activations=np.arange(n * 4, dtype=np.float32).reshape(n, 4),
        beliefs=beliefs,
        diagnostic_beliefs=beliefs.copy(),
        tokens=np.arange(n, dtype=np.int64) % 3,
        previous_tokens=np.full(n, -1, dtype=np.int64),
        env_indices=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        episode_steps=np.array([0, 0, 1, 1, 2, 2, 0, 3, 1, 4]),
        states=np.zeros(n, dtype=np.int64),
        actions=np.array([[0], [1], [0], [1], [0], [1], [2], [1], [2], [1]]),
        rewards=np.zeros(n),
    )


def test_legacy_checkpoint_task_path_resolves_to_current_class():
    assert LegacyActionSymmetryTask is ActionSymmetryTask


def test_checkpoint_loads_only_local_inference_module(tmp_path, monkeypatch):
    environment = HMMEnv(environment_config(1))
    try:
        module = TransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            inference_only=False,
            model_config=dict(BASE_MODEL_CONFIG),
        )
    finally:
        environment.close()
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            parameter.fill_(index / 100.0)
    expected_state = {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }
    module_path = tmp_path.joinpath(*probe_experiment._MODULE_COMPONENTS)
    module.save_to_path(module_path)

    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.rl_module.rl_module import RLModule

    def fail_algorithm_restore(*args, **kwargs):
        raise AssertionError("checkpoint constructors must not run")

    monkeypatch.setattr(Algorithm, "from_checkpoint", fail_algorithm_restore)
    monkeypatch.setattr(RLModule, "from_checkpoint", fail_algorithm_restore)
    with probe_experiment._load_local_inference_checkpoint(tmp_path) as restored:
        assert isinstance(restored.module, TransformerModel)
        assert restored.module.inference_only is True
        assert restored.config.env is HMMEnv
        assert restored.config.num_env_runners == 0
        assert restored.config.num_learners == 0
        for name, value in restored.module.state_dict().items():
            torch.testing.assert_close(value, expected_state[name])


def test_exact_next_token_probabilities_condition_on_action():
    beliefs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    transitions = np.stack(
        [
            np.eye(3),
            np.roll(np.eye(3), shift=1, axis=1),
            np.full((3, 3), 1.0 / 3.0),
        ]
    )
    probabilities = exact_next_token_probabilities(
        beliefs,
        np.array([0, 1]),
        transitions,
        np.eye(3),
    )
    np.testing.assert_allclose(probabilities[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(probabilities[1], [0.0, 1.0, 0.0])


def test_sequence_dataset_stays_within_environment_and_episode():
    data = _probe_data()
    transitions = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    dataset = build_sequence_dataset(
        data,
        context_len=2,
        transition_matrices=transitions,
        emission_matrix=np.eye(3),
    )

    assert len(dataset) == 4
    np.testing.assert_array_equal(
        dataset.embeddings[:, :, 0],
        np.array(
            [
                [0, 8],
                [4, 12],
                [12, 20],
                [20, 28],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(dataset.target_tokens, [1, 2, 1, 0])


def test_transformer_probe_detaches_trunk_and_blind_mode_ignores_action():
    torch.manual_seed(7)
    probe = NextTokenTransformerProbe(
        embedding_dim=4,
        context_len=2,
        condition_on_action=False,
        d_model=8,
        n_heads=2,
        n_layers=1,
    )
    embeddings = torch.randn(3, 2, 4, requires_grad=True)
    actions = torch.tensor([0, 1, 2])
    logits = probe(embeddings, actions)
    alternate = probe(embeddings, torch.tensor([2, 0, 1]))
    torch.testing.assert_close(logits, alternate)
    logits.sum().backward()

    assert embeddings.grad is None
    assert any(parameter.grad is not None for parameter in probe.parameters())


def test_fit_probe_runs_soft_target_training_end_to_end():
    rng = np.random.default_rng(11)

    def dataset(n: int) -> SequenceDataset:
        embeddings = rng.normal(size=(n, 2, 4)).astype(np.float32)
        labels = (embeddings[:, -1, 0] > 0.0).astype(np.int64)
        probabilities = np.full((n, 3), 0.05, dtype=np.float32)
        probabilities[np.arange(n), labels] = 0.9
        return SequenceDataset(
            embeddings=embeddings,
            actions=np.arange(n, dtype=np.int64) % 3,
            target_probabilities=probabilities,
            target_tokens=labels,
        )

    metrics = fit_probe(
        dataset(64),
        dataset(32),
        dataset(32),
        condition_on_action=True,
        device="cpu",
        seed=17,
        config=ProbeTrainingConfig(
            d_model=8,
            n_heads=2,
            n_layers=1,
            batch_size=16,
            max_epochs=3,
            patience=2,
        ),
    )

    assert metrics["stop_gradient"] is True
    assert metrics["n_train"] == 64
    assert metrics["n_validation"] == 32
    assert metrics["n_test"] == 32
    assert np.isfinite(metrics["soft_kl_nats"])


def test_comparison_joins_only_checkpoints_with_belief_mse(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "protocol": {"seed": 42},
        "checkpoints": [
            {
                "checkpoint_name": "initial_checkpoint",
                "agent_steps": 0,
                "belief_probe": {"mse": 0.1, "global_mse_ratio": 0.2},
                "conditions": [
                    {
                        "context_len": 1,
                        "condition_on_selected_action": False,
                        "soft_kl_nats": 0.03,
                        "fraction_predictive_kl_removed": 0.8,
                    }
                ],
            },
            {
                "checkpoint_name": "checkpoint_000002",
                "agent_steps": 99,
                "belief_probe": None,
                "conditions": [],
            },
        ],
    }
    (run_dir / "next_token_probe_curve.json").write_text(json.dumps(payload))

    rows = load_joined_rows(tmp_path)
    assert rows == [
        {
            "seed": 42,
            "checkpoint_name": "initial_checkpoint",
            "agent_steps": 0,
            "condition": "context_1_action_blind",
            "belief_mse": 0.1,
            "belief_global_mse_ratio": 0.2,
            "next_token_soft_kl_nats": 0.03,
            "next_token_fraction_predictive_kl_removed": 0.8,
        }
    ]
    assert summarize(rows)["joined_row_count"] == 1
