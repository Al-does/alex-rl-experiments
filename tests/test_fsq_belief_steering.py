import numpy as np
import pytest
import torch

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_followup import (
    BeliefMap, ResidualSites, config, fit_map, orthogonal_direction, select_lambda, shift,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_prediction import (
    BeliefPolicy, EMISSION, LAMBDAS, TRANSITIONS, simulate, target,
)
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import ReinforceTransformerModel
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def test_targets():
    b = np.array([[.2, .5, .3], [.7, .1, .2], [.3, .3, .4]])
    p = target(b, "point")
    np.testing.assert_allclose(p, [[.7, 0, .3], [0, .8, .2], [0, .6, .4]])
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_allclose(p[:, 2], b[:, 2])
    m = target(b, "mirror")
    np.testing.assert_allclose(target(m, "mirror"), b)
    np.testing.assert_allclose(m[:, 2], b[:, 2])
    assert target(np.empty((0, 3)), "point").shape == (0, 3)
    assert target(np.array([.2, .5, .3]), "point").shape == (3,)
    for invalid in (np.ones(2), np.array([0, np.nan, 1])):
        with pytest.raises(ValueError):
            target(invalid, "point")
    with pytest.raises(ValueError):
        target(b, "bad")
    # Unconstrained affine decoding is intentionally not projected onto the simplex.
    np.testing.assert_allclose(target(np.array([-.1, 1.2, -.1]), "point"), [1.1, 0, -.1])


def test_maps_and_null():
    rng = np.random.default_rng(1)
    b = rng.dirichlet(np.ones(3), 300)
    x = b[:, :2] @ rng.normal(size=(2, 8))+rng.normal(size=8)
    mapping = fit_map(x, b)
    np.testing.assert_allclose(x @ mapping.decoder+mapping.bias, b[:, :2], atol=1e-12)
    for mode in ("point", "mirror"):
        changed = x + shift(x, mapping, mode)
        np.testing.assert_allclose(changed @ mapping.decoder+mapping.bias,
                                   target(b, mode)[:, :2], atol=1e-12)
    null = orthogonal_direction(mapping, 9)
    np.testing.assert_allclose(null @ mapping.decoder, 0, atol=1e-12)
    assert np.linalg.norm(null) == pytest.approx(1)
    sweep = {str(l): {"occupancy": .3-l*.2} for l in LAMBDAS}
    assert select_lambda(sweep, .2) == .5
    assert select_lambda(sweep, 0) == 1.


def test_prediction_and_environment_contract():
    env = HMMEnv(config(16))
    try:
        obs, info = env.reset(seed=3)
        assert obs.shape == (6,)
        b = np.ones(3)/3 * EMISSION[:, obs[:3].argmax()]
        b /= b.sum()
        np.testing.assert_allclose(info["belief_current"], b)
        for action in (0, 1, 2)*3:
            decision = env.task.resolve_action(action, info["state_current"], env.model)
            np.testing.assert_allclose(decision.transition_matrix, TRANSITIONS[action])
            obs, _, _, _, info = env.step(action)
            b = (b @ TRANSITIONS[action])*EMISSION[:, obs[:3].argmax()]
            b /= b.sum()
            np.testing.assert_allclose(info["belief_current"], b, atol=1e-12)
    finally:
        env.close()
    policy = BeliefPolicy(n=12, iterations=100)
    np.testing.assert_array_equal(policy(policy.grid), policy.actions)
    p = simulate(policy, "point", 0, episodes=3, horizon=20)
    assert p == simulate(policy, "mirror", 0, episodes=3, horizon=20)
    assert 0 <= p["occupancy"] <= 1
    assert sum(p["action_fractions"]) == pytest.approx(1)
    for kwargs in ({"episodes": 0}, {"horizon": 0}, {"strength": 2}):
        args = dict(episodes=2, horizon=10, strength=.5)
        args.update(kwargs)
        with pytest.raises(ValueError):
            simulate(policy, "point", **args)


def test_residual_hook_and_cache_contract():
    torch.manual_seed(7)
    env = HMMEnv(config(16))
    module = RLModuleSpec(module_class=ReinforceTransformerModel,
                          observation_space=env.observation_space, action_space=env.action_space,
                          model_config={"d_model": 64, "n_layers": 4, "n_heads": 1,
                                        "context_len": 32}).build().eval()
    obs = torch.as_tensor(np.stack([env.reset(seed=4)[0]]*2))
    env.close()
    state = _initial_state(module, 2, torch.device("cpu"))
    original = {k: v.clone() for k, v in state.items()}
    with torch.inference_mode(), ResidualSites(module) as sites:
        h, base_state = module.encode_step(obs, state)
        np.testing.assert_allclose(module.encoder.final_norm(sites.activations[3]).numpy(), h.numpy(), atol=1e-6)
        for k in state:
            assert torch.equal(state[k], original[k])
        sites.mapping = BeliefMap(np.zeros((64, 2)), np.array([.1, .6]),
                                 np.stack((np.arange(64), np.zeros(64))))
        sites.strength = .5
        sites.site = 0
        _, steered_state = module.encode_step(obs, state)
        assert torch.equal(base_state["kv_k"][:, 0], steered_state["kv_k"][:, 0])
        assert not torch.equal(base_state["kv_k"][:, 1], steered_state["kv_k"][:, 1])
        sites.site = 3
        _, final_state = module.encode_step(obs, state)
        for key in base_state:
            assert torch.equal(base_state[key], final_state[key])
