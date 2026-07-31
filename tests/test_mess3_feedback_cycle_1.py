"""Scientific and wiring tests for guess-driven MESS3 feedback, cycle 1."""

from __future__ import annotations

import numpy as np
import pytest

from envs.hmm import HMMEnv
from envs.mess3.model import (
    CONTROL_TRANSITION_MATRIX,
    PASSIVE_TRANSITION_MATRIX,
    emission_matrix,
    passive_model,
)
from experiments.mess3_feedback_cycle_1 import composition, dynamics
from experiments.mess3_feedback_cycle_1.analysis import (
    CONTEXT_LENGTH,
    _fit_target,
    effective_dimension,
    readout_subspace,
    subspace_overlap,
)
from experiments.mess3_feedback_cycle_1.probe import (
    FeedbackProbeData,
    collect_feedback_probe_data,
    make_feedback_filters,
)
from experiments.mess3_feedback_cycle_1.shared import (
    BASE_MODEL_CONFIG,
    CONDITIONS,
    TOTAL_ENV_STEPS,
    build_config,
    condition_by_name,
    env_config,
)
from experiments.mess3_feedback_cycle_1.task import FeedbackTokenGuessTask
from experiments.mess3_token_guess_cycle_2.analysis import (
    bayesian_optimal_accuracy,
)
from experiments.mess3_token_guess_cycle_2.model import PaperActorCriticModel
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models import TransformerModel


EMISSION = emission_matrix(0.85)


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_guess_kernels_are_stochastic_and_distinct_per_guess():
    for strength in (0.0, 0.35, 0.7, 1.0):
        kernels = dynamics.feedback_transitions(strength)
        assert kernels.shape == (3, 3, 3)
        np.testing.assert_allclose(kernels.sum(axis=-1), 1.0)
        assert (kernels >= 0.0).all()
        # Guess zero is always the identity shift.
        np.testing.assert_allclose(kernels[0], PASSIVE_TRANSITION_MATRIX)
    zero = dynamics.feedback_transitions(0.0)
    assert np.allclose(zero[0], zero[1]) and np.allclose(zero[1], zero[2])
    strong = dynamics.feedback_transitions(0.7)
    assert not np.allclose(strong[0], strong[1])
    assert not np.allclose(strong[1], strong[2])
    full = dynamics.feedback_transitions(1.0)
    np.testing.assert_allclose(
        full[1],
        PASSIVE_TRANSITION_MATRIX @ dynamics.cyclic_shift_matrix(1),
    )
    with pytest.raises(ValueError, match="strength"):
        dynamics.feedback_transitions(1.5)


def test_shift_operators_commute_with_the_circulant_base():
    assert dynamics.is_circulant(PASSIVE_TRANSITION_MATRIX)
    assert dynamics.is_circulant(EMISSION)
    shift = dynamics.feedback_shift_operator(2, 0.4)
    np.testing.assert_allclose(
        PASSIVE_TRANSITION_MATRIX @ shift,
        shift @ PASSIVE_TRANSITION_MATRIX,
    )
    assert not dynamics.is_circulant(CONTROL_TRANSITION_MATRIX)
    with pytest.raises(ValueError, match="circulant"):
        dynamics.feedback_transition(1, 0.5, base=CONTROL_TRANSITION_MATRIX)


def test_factored_lift_lumps_exactly_onto_the_executed_process():
    """The (m, Phi) product must aggregate to the executed 3-state process."""

    strength = 0.7
    lump = dynamics.lumping_matrix()
    joint_emission = dynamics.joint_emission(EMISSION)
    for guess in range(3):
        executed = dynamics.feedback_transition(guess, strength)
        aggregated = dynamics.joint_transition(guess, strength) @ lump
        for chain in range(3):
            for register in range(3):
                index = chain * 3 + register
                state = (chain + register) % 3
                np.testing.assert_allclose(aggregated[index], executed[state])
                np.testing.assert_allclose(joint_emission[index], EMISSION[state])


def test_joint_filter_marginals_are_a_product_state_only_at_the_endpoints():
    """kappa interpolates between lossless and maximally lossy factoring."""

    losses = {}
    for strength in (0.0, 0.35, 0.7, 1.0):
        rollout = composition.simulate_closed_loop(
            strength,
            n_chains=48,
            n_steps=320,
            burn_in=128,
            seed=11,
            record_joint=True,
        )
        losses[strength] = composition.factorization_report(rollout)
    assert losses[0.0]["executed_product_mse"] < 1e-12
    assert losses[1.0]["executed_product_mse"] < 1e-12
    assert losses[0.0]["register_entropy_nats"] < 1e-9
    assert losses[1.0]["register_entropy_nats"] < 1e-9
    for strength in (0.35, 0.7):
        assert losses[strength]["executed_product_mse"] > 1e-2
        # Only the sum of the two factors is ever observed, so each marginal
        # stays at its uniform prior while the joint stays informative.
        assert losses[strength]["register_entropy_nats"] == pytest.approx(
            np.log(3.0),
            abs=1e-6,
        )


def test_ceiling_matches_the_passive_optimum_at_both_endpoints():
    """A deterministic rotation is the passive process in a rotating frame."""

    exact = bayesian_optimal_accuracy(context_length=CONTEXT_LENGTH)
    for strength in (0.0, 1.0):
        ceiling = composition.myopic_ceiling(
            strength,
            context_length=CONTEXT_LENGTH,
            n_chains=192,
            n_steps=768,
            seed=4,
        )
        assert ceiling["accuracy"] == pytest.approx(exact, abs=4e-3)
    interior = composition.myopic_ceiling(
        0.35,
        context_length=CONTEXT_LENGTH,
        n_chains=192,
        n_steps=768,
        seed=4,
    )
    assert 1 / 3 < interior["accuracy"] < exact - 0.05


def test_block_distributions_agree_between_exact_and_empirical_counts():
    rollout = composition.simulate_closed_loop(
        0.0,
        policy="uniform",
        n_chains=256,
        n_steps=1_024,
        burn_in=32,
        seed=9,
    )
    empirical = composition.empirical_block_distribution(
        rollout.tokens,
        length=3,
        n_tokens=3,
    )
    exact = composition.block_distribution(
        PASSIVE_TRANSITION_MATRIX,
        EMISSION,
        length=3,
    )
    assert composition.total_variation(empirical, exact) < 0.01
    np.testing.assert_allclose(exact.sum(), 1.0)


def test_stacked_hmm_is_exact_without_feedback_and_fails_with_it():
    """The marginalized kernel only describes the loop when guesses are inert."""

    inert = composition.single_hmm_report(
        0.0,
        policy="myopic_argmax",
        n_chains=192,
        n_steps=1_024,
        seed=13,
    )
    assert inert["block_tv_marginal_hmm"] <= inert["block_tv_sampling_floor"]
    assert inert["belief_mse_marginal_vs_exact"] < 1e-3

    deterministic = composition.single_hmm_report(
        1.0,
        policy="myopic_argmax",
        n_chains=192,
        n_steps=1_024,
        seed=13,
    )
    assert (
        deterministic["block_tv_marginal_hmm"]
        > 4.0 * deterministic["block_tv_sampling_floor"]
    )


def test_marginalized_transition_stacks_and_renormalizes_the_kernels():
    transitions = dynamics.feedback_transitions(0.7)
    states = np.array([0, 0, 1, 1, 2, 2])
    actions = np.array([1, 1, 2, 2, 0, 0])
    guess_given_state, marginal = composition.marginalized_transition(
        states,
        actions,
        transitions,
        prior_count=0.0,
    )
    np.testing.assert_allclose(guess_given_state.sum(axis=1), 1.0)
    np.testing.assert_allclose(marginal.sum(axis=1), 1.0)
    np.testing.assert_allclose(marginal[0], transitions[1][0])
    np.testing.assert_allclose(marginal[1], transitions[2][1])
    np.testing.assert_allclose(marginal[2], transitions[0][2])


def test_task_rewards_the_current_token_and_executes_the_guessed_kernel():
    model = passive_model(alpha=0.85)
    task = FeedbackTokenGuessTask(model=model, feedback_strength=0.7)
    decision = task.resolve_action(2, state=0, model=model)
    np.testing.assert_allclose(
        decision.transition_matrix,
        dynamics.feedback_transition(2, 0.7),
    )
    np.testing.assert_allclose(task.encode_action(2), [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="feedback_strength"):
        FeedbackTokenGuessTask(model=model, feedback_strength=-0.1)

    condition = condition_by_name("strong_feedback")
    environment = HMMEnv(
        {
            **env_config(condition),
            "episode_length": 4,
            "diagnostics": {"tokens": True, "transitions": True},
        }
    )
    try:
        _, info = environment.reset(seed=5)
        expected = info["raw_token_current"]
        observation, reward, _, _, step_info = environment.step(expected)
        assert reward == 1.0
        assert step_info["raw_token_before"] == expected
        # Token features precede the previous-guess one hot.
        assert observation[expected] == 1.0
        assert observation[3 + expected] == 1.0
        np.testing.assert_allclose(
            step_info["executed_transition_matrix"],
            dynamics.feedback_transition(expected, 0.7),
        )
    finally:
        environment.close()


def test_blind_condition_hides_the_previous_guess_from_the_observation():
    blind = HMMEnv(env_config(condition_by_name("strong_feedback_blind")))
    sighted = HMMEnv(env_config(condition_by_name("strong_feedback")))
    try:
        assert blind.observation_space.shape == (3,)
        assert sighted.observation_space.shape == (6,)
        assert blind.config.observation.action is None
        assert sighted.config.observation.action is not None
    finally:
        blind.close()
        sighted.close()


def test_probe_targets_track_the_environment_and_separate_by_guess():
    condition = condition_by_name("full_feedback")
    environment_config = {
        **env_config(condition),
        "episode_length": 6,
        "diagnostics": {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        },
    }

    def make_environment():
        return HMMEnv(environment_config)

    environment = make_environment()
    try:
        filters = make_feedback_filters(environment, feedback_strength=1.0)
        module = TransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config={
                "context_len": 4,
                "d_model": 24,
                "n_layers": 1,
                "n_heads": 3,
                "max_seq_len": 3,
            },
        )
    finally:
        environment.close()

    marginal = dynamics.feedback_transitions(1.0).mean(axis=0)
    data = collect_feedback_probe_data(
        module,
        make_environment,
        filters.with_marginal(marginal),
        n_steps=48,
        seed=7,
        policy_mode="random",
        n_envs=4,
        warmup=1,
    )
    assert data.activations.shape == (48, 24)
    for target in ("executed", "blind", "marginal", "factor_m", "factor_phi"):
        values = data.target(target)
        assert values.shape == (48, 3)
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-12)
    assert data.joint.shape == (48, 9)
    np.testing.assert_allclose(data.joint.sum(axis=1), 1.0, atol=1e-12)

    # The action-conditioned filter must reproduce the environment exactly,
    # and the action-blind filter must not.
    np.testing.assert_allclose(data.executed, data.diagnostic, atol=1e-12)
    assert np.abs(data.blind - data.executed).max() > 1e-3

    # A deterministic rotation makes the register a delta, so the joint
    # belief is exactly a product state.
    gaps = data.product_state_gap()
    assert gaps["joint_product_mse"] < 1e-20
    assert data.factor_phi.max(axis=1).min() == pytest.approx(1.0)


def test_partial_feedback_pushes_the_joint_belief_off_the_product_manifold():
    condition = condition_by_name("weak_feedback")
    environment_config = {
        **env_config(condition),
        "episode_length": 8,
        "diagnostics": {
            "state": True,
            "belief": True,
            "tokens": True,
            "transitions": True,
        },
    }

    def make_environment():
        return HMMEnv(environment_config)

    environment = make_environment()
    try:
        filters = make_feedback_filters(environment, feedback_strength=0.35)
        module = TransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config={
                "context_len": 4,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 2,
                "max_seq_len": 3,
            },
        )
    finally:
        environment.close()

    data = collect_feedback_probe_data(
        module,
        make_environment,
        filters,
        n_steps=64,
        seed=3,
        policy_mode="random",
        n_envs=4,
        warmup=2,
    )
    assert data.marginal is None
    np.testing.assert_allclose(data.executed, data.diagnostic, atol=1e-12)
    assert data.product_state_gap()["joint_product_mse"] > 1e-4


def _oracle_probe_data(
    activations: np.ndarray,
    executed: np.ndarray,
    blind: np.ndarray,
) -> FeedbackProbeData:
    count = len(executed)
    zeros = np.zeros(count, dtype=np.int64)
    return FeedbackProbeData(
        activations=activations,
        executed=executed,
        diagnostic=executed,
        blind=blind,
        marginal=None,
        joint=np.zeros((count, 9)),
        factor_m=np.zeros((count, 3)),
        factor_phi=np.zeros((count, 3)),
        tokens=zeros,
        previous_tokens=zeros,
        actions=zeros,
        previous_actions=zeros,
        states=zeros,
        env_indices=zeros,
        episode_steps=np.arange(count, dtype=np.int64),
        rewards=np.zeros(count),
    )


def test_action_awareness_ratio_separates_aware_from_blind_features():
    """A representation carrying one belief must not decode the other."""

    rollout = composition.simulate_closed_loop(
        0.7,
        n_chains=32,
        n_steps=288,
        burn_in=32,
        seed=21,
    )
    executed = rollout.beliefs.reshape(-1, 3)
    blind = composition.hmm_filter(
        rollout.tokens,
        PASSIVE_TRANSITION_MATRIX,
        EMISSION,
    ).reshape(-1, 3)
    # The two targets must actually disagree, or the contrast is vacuous.
    assert np.square(executed - blind).mean() > 1e-2
    half = len(executed) // 2

    ratios = {}
    for name, features in (("aware", executed), ("blind", blind)):
        train = _oracle_probe_data(
            features[:half],
            executed[:half],
            blind[:half],
        )
        test = _oracle_probe_data(
            features[half:],
            executed[half:],
            blind[half:],
        )
        fitted = {
            target: _fit_target(train, test, target)[0]
            for target in ("executed", "blind")
        }
        ratios[name] = (
            fitted["executed"]["global_mse_ratio"]
            / fitted["blind"]["global_mse_ratio"]
        )
    assert ratios["aware"] < 0.1
    assert ratios["blind"] > 10.0


def test_effective_dimension_and_subspace_overlap_report_known_geometry():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(12, 12)))[0]
    coefficients = rng.normal(size=(400, 3)) * np.array([5.0, 4.0, 1e-6])
    activations = coefficients @ basis[:, :3].T
    report = effective_dimension(activations, variance_fraction=0.95)
    assert report["effective_dimension"] == 2

    first = readout_subspace(basis[:, :2], rank=2)
    second = readout_subspace(basis[:, 2:4], rank=2)
    assert subspace_overlap(first, second) == pytest.approx(0.0, abs=1e-12)
    assert subspace_overlap(first, first) == pytest.approx(1.0)


def test_every_condition_builds_a_fresh_gamma_zero_ppo_recipe(tmp_path):
    context = _context(tmp_path)
    assert [condition.name for condition in CONDITIONS] == [
        "no_feedback",
        "weak_feedback",
        "strong_feedback",
        "full_feedback",
        "strong_feedback_blind",
    ]
    assert TOTAL_ENV_STEPS == 2_500_000
    assert BASE_MODEL_CONFIG["context_length"] == CONTEXT_LENGTH

    strengths = {}
    for condition in CONDITIONS:
        first = build_config(context, condition.name)
        second = build_config(context, condition.name)
        assert first is not second
        assert first.gamma == 0.0
        assert first.lambda_ == 0.0
        assert first.num_env_runners == 0
        assert first.train_batch_size_per_learner == 2_048
        assert first.minibatch_size == 256
        assert first.vf_loss_coeff == 0.5
        assert first.rl_module_spec.module_class is PaperActorCriticModel
        environment = first.env_config
        assert environment["delay"] == 1
        assert environment["task"]["kwargs"]["feedback_strength"] == (
            condition.feedback_strength
        )
        strengths[condition.name] = condition.feedback_strength
    assert strengths == {
        "no_feedback": 0.0,
        "weak_feedback": 0.35,
        "strong_feedback": 0.7,
        "full_feedback": 1.0,
        "strong_feedback_blind": 0.7,
    }
    with pytest.raises(ValueError, match="unknown feedback condition"):
        build_config(context, "missing")


def test_single_gpu_profile_reserves_cuda_for_the_learner(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cuda4090_gpuinfer"],
    )
    config = build_config(context, "strong_feedback")
    assert config.num_gpus_per_learner == 1
    assert config.num_gpus_per_env_runner == 0
