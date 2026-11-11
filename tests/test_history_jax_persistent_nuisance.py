from dataclasses import asdict
from pathlib import Path
import subprocess
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import env
from experiments.belief_quotient_history_jax_2026_11 import shared as old_ladder
from experiments.reward_factors_history_jax_2026_11 import shared as old_factors
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as ladder
from experiments.belief_quotient_persistent_nuisance_jax_2026_11.design import (
    controller_references, history_belief,
)
from experiments.reward_factors_persistent_nuisance_jax_2026_11 import shared as factors


def test_scalar_filter_random_histories():
    process = ladder.ladder_process(2)
    rng = np.random.default_rng(9)
    belief = env.belief_reset(process, jnp.int32(0))
    for _ in range(60):
        for action in range(3):
            rho = float(belief[2])
            u, v = np.asarray(process.transitions[action, [0, 2], 2])
            r = u * (1 - rho) + v * rho
            for split in (.02, .37, .98):
                fine = jnp.array([(1 - rho) * split, (1 - rho) * (1 - split), rho])
                probabilities = np.asarray(fine @ process.transitions[action] @ process.emission)
                assert probabilities[:2].sum() == pytest.approx(.25 * (1 - r), abs=1e-7)
                assert probabilities[2] == pytest.approx(r + .75 * (1 - r), abs=2e-7)
                for token in range(3):
                    posterior = env.belief_update(process, fine, action, token)
                    assert np.isfinite(posterior).all()
                    assert posterior[2] == pytest.approx(
                        0 if token < 2 else r / (r + .75 * (1 - r)), abs=1e-7)
        belief = env.belief_update(process, belief, int(rng.integers(3)), int(rng.integers(3)))


def test_asymmetric_history_targets_and_reset_identity_counterexample():
    expected = ([.792774050481, .074363078022, .132862871497],
                [.295147599840, .571989528662, .132862871497])
    process = ladder.ladder_process(2)
    baseline = ladder.BASELINE_TRANSITION_MATRIX.copy()
    baseline[:2] = [.4875, .4875, .025]
    kernels = np.stack([baseline] + [
        ladder.tilt_reward_state(baseline, ladder.DIRECTIONS[2][:, a]) for a in (1, 2)])
    counterexamples = []
    for tokens, target in zip(([0, 0, 0, 2], [1, 1, 0, 2]), expected):
        belief = env.belief_reset(process, tokens[0])
        for token in tokens[1:]:
            belief = env.belief_update(process, belief, 1, token)
        np.testing.assert_allclose(belief, target, atol=1e-7)
        np.testing.assert_allclose(history_belief(tokens), target, atol=1e-12)
        counterexamples.append(history_belief(tokens, kernels))
        assert not np.allclose(counterexamples[-1], target)
    np.testing.assert_allclose(*counterexamples, atol=1e-12)


def test_stationary_references():
    result = controller_references()
    assert result["current_token_policy"] == (1, 1, 0)
    assert result["current_token"] == pytest.approx(.228861871656, abs=1e-12)
    assert result["feasible_three_token"] == pytest.approx(.251424641015, abs=1e-12)
    assert result["relative_gain"] == pytest.approx(.098586842778, abs=1e-12)
    fractions = result["c_action_fractional_rewards"]
    assert np.argmax(fractions) == 0
    assert max(fractions) == pytest.approx(result["current_token"], abs=1e-12)
    kernels = ladder.action_transitions(2)
    u, v = kernels[:, 0, 2], kernels[:, 2, 2]
    rng = np.random.default_rng(3)
    for weights in rng.dirichlet(np.ones(3), 50):
        U = .25 * u[1] + .75 * (weights @ u)
        assert U / (U + 1 - weights @ v) <= max(fractions) + 1e-12


def test_factor_kernels_and_filter():
    first = ladder.ladder_process(3)
    joint = factors.factor_process("reward_both")
    other = factors.factor_process("reward_factor_1")
    for name in ("initial", "transitions", "emission", "reward_components"):
        np.testing.assert_array_equal(getattr(joint, name), getattr(other, name))
    np.testing.assert_array_equal(joint.reward, joint.reward_components.sum(0))
    np.testing.assert_array_equal(other.reward, other.reward_components[0])
    np.testing.assert_allclose(joint.emission, np.kron(first.emission, first.emission))
    for action, (a, b) in enumerate(factors.ACTION_PAIRS):
        np.testing.assert_allclose(joint.transitions[action],
                                   np.kron(first.transitions[a], first.transitions[b]), atol=1e-7)
    beliefs = [env.belief_reset(first, 0), env.belief_reset(first, 2)]
    belief = env.belief_reset(joint, 2)
    rng = np.random.default_rng(4)
    for _ in range(60):
        action, token = int(rng.integers(9)), int(rng.integers(9))
        for factor, (a, t) in enumerate(zip(factors.ACTION_PAIRS[action], divmod(token, 3))):
            beliefs[factor] = env.belief_update(first, beliefs[factor], a, t)
        belief = env.belief_update(joint, belief, action, token)
        np.testing.assert_allclose(belief, np.kron(*beliefs), atol=2e-7)
        assert np.isfinite(belief).all()


def test_zero_emissions_and_actor_inputs():
    process = ladder.ladder_process(2)
    keys = jax.random.split(jax.random.PRNGKey(5), 4096)
    tokens = jax.vmap(lambda key: env._categorical(key, process.emission[2]))(keys)
    assert np.all(np.asarray(tokens) == 2)
    assert np.array_equal(process.emission[2], [0, 0, 1])
    for token in range(3):
        assert np.isfinite(env.belief_reset(process, token)).all()
    state = env.EnvState(jnp.int32(2), jnp.int32(2), jnp.int32(-1))
    np.testing.assert_array_equal(env.observe(process, state), [0, 0, 1, 0, 0, 0])


@pytest.mark.parametrize("new,old,condition", [
    (ladder, old_ladder, 2), (factors, old_factors, "reward_both"),
])
def test_recipe_inheritance(new, old, condition):
    assert new.TOTAL_ENV_STEPS == old.TOTAL_ENV_STEPS == 200_000_000
    for smoke in (False, True):
        assert asdict(new.train_config(smoke)) == asdict(old.train_config(smoke))
        assert new.train_config(smoke).temperature == 1.
    process = (new.ladder_process(condition) if new is ladder else new.factor_process(condition))
    for architecture in new.ARCHITECTURES:
        assert new.model_spec(architecture, process) == old.model_spec(architecture, process)
        with patch.object(new, "run_study", side_effect=lambda *args, **kwargs: kwargs):
            kwargs = {"variant" if new is ladder else "condition": condition}
            result = new.run_condition(type("Context", (), {"smoke": False})(),
                                       architecture=architecture, **kwargs)
        metadata = result["recipe_extra"]
        assert metadata["source_study"] == old.STUDY
        assert metadata["environment_revision"] == "persistent_nuisance_v1"
        np.testing.assert_array_equal(metadata["emission_matrix"], ladder.EMISSION_MATRIX)
        np.testing.assert_array_equal(metadata["baseline_transition_matrix"],
                                      ladder.BASELINE_TRANSITION_MATRIX)
        assert metadata["effect_size"] == 1.5
        assert metadata["directions"] == {str(k): v.tolist() for k, v in old_ladder.DIRECTIONS.items()}
        assert result["representation"] == ("embedding" if new is ladder else "residual")
    with pytest.raises(ValueError):
        if new is ladder:
            new.ladder_process(0)
        else:
            new.factor_process("unknown")


def test_historical_studies_untouched():
    root = Path(__file__).resolve().parents[1]
    paths = ["experiments/belief_quotient_history_jax_2026_11",
             "experiments/reward_factors_history_jax_2026_11",
             "experiments/belief_quotient_history_jax_lumped_2026_11",
             "experiments/reward_factors_history_jax_lumped_2026_11"]
    assert subprocess.check_output(["git", "diff", "HEAD", "--", *paths], cwd=root) == b""


def test_epsilon_greedy_policy_log_probs():
    from experiments.belief_quotient_history_jax_2026_11.reinforce import policy_log_probs

    logits = jnp.array([[2.0, -1.0, 0.5], [0.0, 0.0, 9.0]])
    assert jnp.array_equal(policy_log_probs(logits, 1.5), jax.nn.log_softmax(logits / 1.5))
    mixed = jnp.exp(policy_log_probs(logits, 1.0, 0.2))
    expected = 0.8 * jax.nn.softmax(logits) + 0.2 / 3
    np.testing.assert_allclose(mixed, expected, rtol=1e-6)
    np.testing.assert_allclose(mixed.sum(-1), 1.0, rtol=1e-6)
