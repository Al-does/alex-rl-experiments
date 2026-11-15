import numpy as np

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_refinement import InterpolatedPolicy


def test_barycentric_reproduces_affine_values():
    policy = InterpolatedPolicy(20, .99)
    rng = np.random.default_rng(52)
    b = np.concatenate((rng.dirichlet(np.ones(3), 500), np.eye(3)))
    indices, weights = policy.stencil(b)
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-14)
    reconstructed = (policy.grid[indices]*weights[..., None]).sum(-2)
    np.testing.assert_allclose(reconstructed, b, atol=1e-14)


def test_symmetric_beliefs_have_stable_action_ties():
    policy = InterpolatedPolicy(20, .995)
    rho = np.linspace(0, 1, 201)
    b = np.stack(((1-rho)/2, (1-rho)/2, rho), -1)
    assert policy.convergence['converged']
    assert np.all(policy(b) != 2)
    assert policy(np.empty((0, 3))).shape == (0,)
