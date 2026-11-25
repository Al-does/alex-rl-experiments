import numpy as np
import pytest

from experiments.reward_factors_rs_trivial_eps02_jax_temp15_ctx32_2026_11 import algorithms, design, process


def test_process_factors_and_reward():
    proc = process.rs_trivial_process()
    first, second, emission = process.factor_matrices()
    assert np.allclose(np.asarray(proc.transitions).sum(-1), 1)
    assert np.allclose(np.asarray(proc.emission), np.kron(emission, emission))
    for index, (a, b) in enumerate(process.ACTION_PAIRS):
        assert np.allclose(np.asarray(proc.transitions)[index], np.kron(first[a], second[b]))
    states = np.divmod(np.arange(9), 3)
    assert np.allclose(np.asarray(proc.reward), (states[0] == 2).astype(float) + (states[1] == 2))
    # Lumpability onto {0,1} vs {2} in both factors.
    for t in (first, second):
        assert np.allclose(t[:, 0, 2], t[:, 1, 2])
    assert emission[0, 2] == emission[1, 2] and emission[0, 0] != emission[1, 0]


def test_targets_shapes():
    belief = np.random.default_rng(0).dirichlet(np.ones(9), 5)
    targets = process.make_targets(lambda f1, f2: f1[:, 2] + f2[:, 2])(belief)
    first, second = process.factor_marginals(belief)
    assert np.allclose(targets["rho_2"][:, 0], second[:, 2])
    assert np.allclose(targets["delta_1"][:, 0], first[:, 0] - first[:, 1])
    assert targets["value_star"].shape == (5, 1)


def test_control_one_dimensional_value_two_dimensional():
    s1, s2 = design.factor_solutions()
    assert (s2["q"].argmax(0) == 1).all()
    assert len(np.unique(s1["q"].argmax(0))) > 1
    rho = np.linspace(0, 1, 11)
    b = design.from_coords(rho, np.zeros_like(rho))
    v2 = design.factor_value(1, b)
    assert np.allclose(np.diff(v2, 2), 0, atol=1e-6) and v2[-1] > v2[0]
    assert np.allclose(design.factor_value(0, design.from_coords(rho, 0.5 + 0 * rho)),
                       design.factor_value(0, design.from_coords(rho, -0.5 + 0 * rho)), atol=1e-8)


@pytest.mark.parametrize("arm", algorithms.ARMS)
def test_backends_configure_four_epochs(arm):
    from experiments.reward_factors_history_jax_2026_11 import shared as source
    config = algorithms.BACKENDS[arm].configure(source.train_config(True))
    assert config.num_epochs == 4
