from __future__ import annotations

import json

import numpy as np
import torch

from experiments.mess3_belief_geometry_2026_07.probe import ProbeData
from experiments.mess3_reward_state_action_symmetry_cycle_5.compare_next_token_probe import (
    load_joined_rows,
    summarize,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.next_token_probe.probe import (
    NextTokenTransformerProbe,
    build_sequence_dataset,
    exact_next_token_probabilities,
)


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
