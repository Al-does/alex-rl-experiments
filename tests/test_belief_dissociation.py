import numpy as np
import pytest
import torch

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import ReinforceTransformerModel
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_followup import config, fit_map
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_prediction import (
    BeliefPolicy, EMISSION, simulate, target, transitions_for,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_pulse import (
    branch_trace, clone_state, decay_rate, exact_trace, intact_history,
)


@pytest.mark.parametrize("mode", ["rho_flip", "rho_point"])
def test_rho_targets(mode):
    b = np.array([[.1, .3, .6], [.6, .2, .2], [0, 0, 1], [-.1, -.2, 1.3]])
    t = target(b, mode)
    expected_rho = 1-b[:, 2] if mode == "rho_flip" else (b[:, 2] < .5)
    np.testing.assert_allclose(t[:, 2], expected_rho)
    np.testing.assert_allclose(t.sum(1), 1)
    np.testing.assert_allclose(t[0, 0]*3, t[0, 1])
    np.testing.assert_allclose(t[1, 0], 3*t[1, 1])
    np.testing.assert_allclose(t[2:, 0], t[2:, 1])
    assert target(np.empty((0, 3)), mode).shape == (0, 3)
    assert target(np.array([.25, .25, .5]), "rho_point")[2] == 0
    with pytest.raises(ValueError):
        target(np.array(1.), mode)
    if mode == "rho_flip":
        np.testing.assert_allclose(target(t[:2], mode), b[:2])


def test_rsq_prediction_contract():
    env = HMMEnv(config(32, 2))
    try:
        for a, matrix in enumerate(transitions_for(2)):
            np.testing.assert_allclose(env.task.transition_matrix_for_action(a), matrix)
            np.testing.assert_allclose(matrix[0, 2], matrix[1, 2])
    finally:
        env.close()
    policy = BeliefPolicy(n=12, iterations=100, variant=2)
    for mode in ("point", "mirror"):
        intact = simulate(policy, mode, 0, episodes=4, horizon=40)
        perturbed = simulate(policy, mode, 1, episodes=4, horizon=40)
        assert intact == perturbed
    with pytest.raises(ValueError):
        transitions_for(1)


def test_decay_fit():
    fit = decay_rate(.2*.8**np.arange(49))
    assert fit["per_step_multiplier"] == pytest.approx(.8)
    assert fit["r2"] == pytest.approx(1)
    assert decay_rate(np.zeros(49))["per_step_multiplier"] is None


@pytest.mark.parametrize("variant", [2, 3])
def test_zero_pulse_cache_and_exact_filter(variant):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    env = HMMEnv(config(32, variant))
    module = RLModuleSpec(module_class=ReinforceTransformerModel,
                          observation_space=env.observation_space, action_space=env.action_space,
                          model_config={"d_model": 64, "n_layers": 4, "n_heads": 1,
                                        "context_len": 32}).build().eval()
    env.close()
    rng = np.random.default_rng(3)
    b = rng.dirichlet(np.ones(3), 200)
    maps = [fit_map(b[:, :2] @ rng.normal(size=(2, 64)), b) for _ in range(5)]
    history = intact_history(module, maps[3], variant, 2, 32, (12,))
    state = history["snapshots"][12]
    old = clone_state(state)
    for site in range(3):
        reading, p = branch_trace(module, maps, history["observations"][12:21], state,
                                  site=site, mode="rho_flip", strength=0)
        np.testing.assert_array_equal(reading, history["readings"][12:21])
        np.testing.assert_array_equal(p, history["probabilities"][12:21])
        for key in state:
            assert torch.equal(state[key], old[key])
    full, coarse = exact_trace(history, variant, 12, 8, "mirror")
    b0 = history["beliefs"][12]
    np.testing.assert_allclose(full[0, :, 0], 0)
    np.testing.assert_allclose(full[0, :, 1], -2*(b0[:, 0]-b0[:, 1]))
    # Independent unperturbed replay must match the environment diagnostics.
    bnext = b0.copy()
    matrices = transitions_for(variant)
    for k in range(1, 9):
        obs = history["observations"][12+k]
        action, token = obs[:, 3:].argmax(1), obs[:, :3].argmax(1)
        bnext = np.einsum("bs,bsj->bj", bnext, matrices[action])*EMISSION[:, token].T
        bnext /= bnext.sum(1, keepdims=True)
        np.testing.assert_allclose(bnext, history["beliefs"][12+k], atol=1e-12)
    if variant == 2:
        np.testing.assert_array_equal(coarse, 0)
    with pytest.raises(ValueError):
        branch_trace(module, maps, history["observations"][12:21], state, site=3, mode="mirror")
