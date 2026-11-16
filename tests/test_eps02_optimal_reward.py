import itertools
import json

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import optimal_reward as ref


def test_power_and_reactive_gradient():
    kernels, emission = ref.eps02.action_transitions(3), ref.eps02.emission_matrix()
    initial, reward = np.full(3, 1/3), np.array([0., 0., 1.])
    policy = np.full((3, 3), 1/3)
    transition = np.einsum("so,oa,asj->sj", emission, policy, kernels)
    distribution, total = initial.copy(), 0.
    for _ in range(9):
        total += distribution @ reward
        distribution = distribution @ transition
    value, derivative = ref.reactive_value(policy, kernels, emission, initial, reward, 9)
    assert value == pytest.approx(total/9)
    numerical = approx_derivative(
        lambda x: ref.reactive_value(x.reshape(3, 3), kernels, emission, initial, reward, 9)[0],
        policy.ravel())
    np.testing.assert_allclose(derivative.ravel(), numerical.ravel(), atol=1e-7)


def test_grid_reproduces_beliefs_and_linear_values():
    grid = ref.SimplexGrid(9)
    beliefs = np.random.default_rng(4).dirichlet(np.ones(3), 100)
    beliefs = np.concatenate([beliefs, np.eye(3), grid.points])
    for coordinate in range(3):
        np.testing.assert_allclose(grid.interpolate(grid.points[:, coordinate], beliefs),
                                   beliefs[:, coordinate], atol=1e-14)
    indices, weights = grid.stencil(beliefs)
    assert indices.min() >= 0 and indices.max() < len(grid.points)
    assert weights.min() >= -1e-14
    probabilities, posterior = ref.successors(grid.points, ref.v2.action_transitions(3),
                                               ref.v2.emission_matrix())
    assert np.isfinite(grid.interpolate(grid.points[:, 2], posterior)).all()
    with pytest.raises(ValueError):
        ref.SimplexGrid(0)


def test_belief_bounds_and_reset_reward():
    kernels, emission = ref.eps02.action_transitions(3), ref.eps02.emission_matrix()
    grid, values = ref.belief_dp(kernels, emission, 2, 12)
    beliefs = np.random.default_rng(7).dirichlet(np.ones(3), 40)
    exact = beliefs[:, 2] + (beliefs @ kernels[:, :, 2].T).max(-1)
    assert np.all(grid.interpolate(values[2], beliefs) >= exact-1e-12)
    report = ref.belief_reference(kernels, emission, 1, 12, 200, 4)
    assert report["upper_bound"] == pytest.approx(1/3)
    assert report["post_warmup64_policy"] is None
    assert report["feasible_policy"]["ci95"][0] < 1/3 < report["feasible_policy"]["ci95"][1]


def test_reactive_search_and_product_policy():
    kernels, emission = ref.eps02.action_transitions(1), ref.eps02.emission_matrix()
    result = ref.reactive_search(kernels, emission, np.full(3, 1/3),
                                np.array([0., 0., 1.]), 8, starts=1)
    expected = ref.reactive_value(np.eye(3)[[1, 1, 1]], kernels, emission,
                                 np.full(3, 1/3), np.array([0., 0., 1.]), 8)[0]
    assert result["best_known"] == pytest.approx(expected, abs=1e-9)
    first = np.eye(3)[[1, 2, 0]]
    joint = ref.product_policy(first, first)
    np.testing.assert_array_equal(joint.sum(-1), np.ones(9))
    assert joint.shape == (9, 9)
    for x, y in itertools.product(range(3), repeat=2):
        action = ref.ACTION_PAIRS[joint[3*x+y].argmax()]
        assert action == (first[x].argmax(), first[y].argmax())


def test_comparisons_empty_and_single_run(tmp_path):
    assert ref.source_comparisons(tmp_path) == []
    directory = tmp_path / "experiments" / ref.LADDER / "variant_2_transformer" / "results" / "run"
    directory.mkdir(parents=True)
    summary = {"seed": 42, "architecture": "transformer", "condition": "variant_2",
               "training": {"final_reward_per_step": .4},
               "checkpoint_reports": [{"checkpoint_step": 20, "policy": {"reward_per_step": .5}},
                                      {"checkpoint_step": 0, "policy": {"reward_per_step": .1}}]}
    (directory / "condition_summary.json").write_text(json.dumps(summary))
    (directory / "training_curves.jsonl").write_text('{"reward_per_step": 0.4}\n')
    record, = ref.source_comparisons(tmp_path)
    assert record["final_steps"] == 20
    assert record["greedy_post_warmup64"] == .5
    assert record["train_reward_per_step"] == .4


def test_cli_output_and_refusal(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(ref, "build_report", lambda *args: {"achieved": []})
    monkeypatch.setattr("sys.argv", ["optimal_reward", "--output", str(output)])
    ref.main()
    assert json.loads(output.read_text()) == {"achieved": []}
    with pytest.raises(SystemExit):
        ref.main()
    output.unlink()
    monkeypatch.setattr("sys.argv", ["optimal_reward", "--output", str(output), "--episodes", "1"])
    with pytest.raises(SystemExit):
        ref.main()
