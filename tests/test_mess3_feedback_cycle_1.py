"""Scientific and wiring tests for guess-driven MESS3 feedback, cycle 1."""

from __future__ import annotations

import numpy as np
import pytest

from envs.hmm import HMMEnv
from envs.mess3.model import (
    CONTROL_TRANSITION_MATRIX,
    PASSIVE_TRANSITION_MATRIX,
    emission_matrix,
)
from experiments.mess3_feedback_cycle_1 import composition, dynamics
from experiments.mess3_feedback_cycle_1.analysis import (
    CONTEXT_LENGTH,
    _fit_target,
    effective_dimension,
    readout_subspace,
    subspace_overlap,
)
from experiments.mess3_feedback_cycle_1.model import composed_model
from experiments.mess3_feedback_cycle_1.probe import (
    FeedbackProbeData,
    collect_feedback_probe_data,
    make_feedback_filters,
)
from experiments.mess3_feedback_cycle_1.shared import (
    BASE_MODEL_CONFIG,
    CONDITIONS,
    OPERATING_STRENGTH,
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
    for strength in (0.0, 0.3, 0.7, 1.0):
        kernels = dynamics.composite_transitions(strength)
        assert kernels.shape == (3, 3, 3)
        np.testing.assert_allclose(kernels.sum(axis=-1), 1.0)
        assert (kernels >= 0.0).all()
        # Guess zero is always the register-inert kernel.
        np.testing.assert_allclose(kernels[0], PASSIVE_TRANSITION_MATRIX)
    inert = dynamics.composite_transitions(0.0)
    assert np.allclose(inert[0], inert[1]) and np.allclose(inert[1], inert[2])
    strong = dynamics.composite_transitions(0.7)
    assert not np.allclose(strong[0], strong[1])
    assert not np.allclose(strong[1], strong[2])
    np.testing.assert_allclose(
        dynamics.composite_transitions(1.0)[1],
        PASSIVE_TRANSITION_MATRIX @ dynamics.cyclic_shift_matrix(1),
    )
    with pytest.raises(ValueError, match="strength"):
        dynamics.composite_transitions(1.5)


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
        dynamics.composite_transition(1, 0.5, base=CONTROL_TRANSITION_MATRIX)


def test_joint_kernel_lumps_exactly_onto_the_composite_process():
    """``(m, phi)`` must aggregate onto ``s = m + phi`` for the scored token."""

    strength = 0.7
    lump = dynamics.lumping_matrix()
    scored = dynamics.composite_likelihood(EMISSION, register_noise=1.0)
    for guess in range(3):
        composite = dynamics.composite_transition(guess, strength)
        aggregated = dynamics.joint_transition(guess, strength) @ lump
        for chain in range(3):
            for register in range(3):
                index = chain * 3 + register
                state = (chain + register) % 3
                np.testing.assert_allclose(aggregated[index], composite[state])
                np.testing.assert_allclose(scored[index], EMISSION[state])


def test_register_channel_makes_the_operator_factor_only_at_zero_noise():
    """Epsilon interpolates between Definition 2.1 and an unfactorable operator."""

    kernel = dynamics.joint_transition(2, 0.7)
    for noise, factors in ((0.0, True), (0.4, False), (1.0, False)):
        paired = dynamics.joint_emission(EMISSION, register_noise=noise)
        # Check whether diag(P(token | m, phi)) splits as A(m) (x) B(phi) for
        # every observable token; the transition already does.
        splits = True
        for token in range(paired.shape[1]):
            grid = paired[:, token].reshape(3, 3)
            rank = np.linalg.matrix_rank(grid, tol=1e-12)
            splits &= rank == 1
        assert bool(splits) is factors, (noise, factors)
        np.testing.assert_allclose(paired.sum(axis=1), 1.0)
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="noise"):
        dynamics.register_channel(1.5)


def test_factoring_is_free_at_zero_noise_and_vacuous_at_one():
    """The cost of a product representation ramps monotonically with epsilon."""

    costs, entropies = {}, {}
    for noise in (0.0, 0.3, 0.85, 1.0):
        rollout = composition.simulate_closed_loop(
            OPERATING_STRENGTH,
            noise,
            n_chains=48,
            n_steps=320,
            burn_in=128,
            seed=11,
        )
        process = composition.composed_process(OPERATING_STRENGTH, noise)
        report = composition.factorization_report(process, rollout)
        costs[noise] = report["factored_cost_nats"]
        entropies[noise] = report["register_entropy_nats"]

    assert costs[0.0] == pytest.approx(0.0, abs=1e-12)
    assert costs[0.0] < costs[0.3] < costs[0.85] < costs[1.0]
    # With a pure-noise report only the factor sum is observable, so the
    # register marginal sits exactly at its uniform prior.
    assert entropies[1.0] == pytest.approx(np.log(3.0), abs=1e-6)
    assert entropies[0.0] < entropies[0.3] < entropies[0.85] < entropies[1.0]


def test_ceiling_matches_the_passive_optimum_at_both_strength_endpoints():
    """A deterministic rotation is the passive process in a rotating frame."""

    exact = bayesian_optimal_accuracy(context_length=CONTEXT_LENGTH)
    for strength in (0.0, 1.0):
        ceiling = composition.myopic_ceiling(
            strength,
            1.0,
            context_length=CONTEXT_LENGTH,
            n_chains=192,
            n_steps=768,
            seed=4,
        )
        assert ceiling["accuracy"] == pytest.approx(exact, abs=5e-3)
    interior = composition.myopic_ceiling(
        OPERATING_STRENGTH,
        1.0,
        context_length=CONTEXT_LENGTH,
        n_chains=192,
        n_steps=768,
        seed=4,
    )
    assert 1 / 3 < interior["accuracy"] < exact - 0.05


def test_register_noise_barely_moves_the_ceiling():
    """Epsilon must isolate factorability from raw task difficulty."""

    ceilings = [
        composition.myopic_ceiling(
            OPERATING_STRENGTH,
            noise,
            context_length=CONTEXT_LENGTH,
            n_chains=192,
            n_steps=768,
            seed=4,
        )["accuracy"]
        for noise in (0.0, 0.3, 0.85, 1.0)
    ]
    assert max(ceilings) - min(ceilings) < 0.02


def test_block_distributions_agree_between_exact_and_empirical_counts():
    rollout = composition.simulate_closed_loop(
        0.0,
        1.0,
        policy="uniform",
        n_chains=256,
        n_steps=1_024,
        burn_in=64,
        seed=9,
    )
    empirical = composition.empirical_block_distribution(
        rollout.scored_tokens,
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
        1.0,
        policy="myopic_argmax",
        n_chains=192,
        n_steps=1_024,
        seed=13,
    )
    assert inert["block_tv_marginal_hmm"] <= inert["block_tv_sampling_floor"]
    deterministic = composition.single_hmm_report(
        1.0,
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
    transitions = dynamics.composite_transitions(0.7)
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


def test_composed_model_is_a_valid_nine_state_hmm():
    model = composed_model(feedback_strength=0.7, register_noise=0.3)
    assert model.n_states == 9 and model.n_tokens == 9
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.emission_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.initial_distribution.sum(), 1.0)
    # The reference kernel is the register-inert product, recoverable exactly.
    np.testing.assert_allclose(
        dynamics.chain_factor(model.transition_matrix),
        PASSIVE_TRANSITION_MATRIX,
    )
    # The register starts pinned at zero.
    np.testing.assert_allclose(model.initial_distribution[[1, 2, 4, 5, 7, 8]], 0.0)


def test_task_scores_the_composite_sub_token_and_executes_the_guessed_kernel():
    model = composed_model(feedback_strength=0.7, register_noise=0.3)
    task = FeedbackTokenGuessTask(model=model, feedback_strength=0.7)
    assert task.action_space.n == 3
    decision = task.resolve_action(2, state=0, model=model)
    np.testing.assert_allclose(
        decision.transition_matrix,
        dynamics.joint_transition(2, 0.7),
    )
    np.testing.assert_allclose(task.encode_action(2), [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="feedback_strength"):
        FeedbackTokenGuessTask(model=model, feedback_strength=-0.1)

    condition = condition_by_name("factoring_costly")
    environment = HMMEnv(
        {
            **env_config(condition),
            "episode_length": 4,
            "diagnostics": {"tokens": True, "transitions": True},
        }
    )
    try:
        _, info = environment.reset(seed=5)
        scored = info["raw_token_current"] // 3
        observation, reward, _, _, step_info = environment.step(scored)
        assert reward == 1.0
        assert step_info["raw_token_before"] // 3 == scored
        # Nine token features precede the three-way previous-guess one hot.
        assert observation[step_info["raw_token_before"]] == 1.0
        assert observation[9 + scored] == 1.0
        np.testing.assert_allclose(
            step_info["executed_transition_matrix"],
            dynamics.joint_transition(scored, condition.feedback_strength),
        )
    finally:
        environment.close()


def test_blind_conditions_hide_the_previous_guess_from_the_observation():
    blind = HMMEnv(env_config(condition_by_name("factoring_impossible_blind")))
    sighted = HMMEnv(env_config(condition_by_name("factoring_impossible")))
    try:
        assert blind.observation_space.shape == (9,)
        assert sighted.observation_space.shape == (12,)
        assert blind.config.observation.action is None
        assert sighted.config.observation.action is not None
    finally:
        blind.close()
        sighted.close()


def _probe_module(environment, width: int = 24):
    return TransformerModel(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        model_config={
            "context_len": 4,
            "d_model": width,
            "n_layers": 1,
            "n_heads": 3,
            "max_seq_len": 3,
        },
    )


def _probe(condition_name: str, *, n_steps: int = 48, marginal: bool = False):
    condition = condition_by_name(condition_name)
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
        filters = make_feedback_filters(
            environment,
            feedback_strength=condition.feedback_strength,
        )
        module = _probe_module(environment)
    finally:
        environment.close()
    if marginal:
        filters = filters.with_marginal(filters.transitions.mean(axis=0))
    return collect_feedback_probe_data(
        module,
        make_environment,
        filters,
        n_steps=n_steps,
        seed=7,
        policy_mode="random",
        n_envs=4,
        warmup=1,
    )


def test_probe_targets_track_the_environment_and_separate_by_guess():
    data = _probe("factoring_free", marginal=True)
    assert data.activations.shape == (48, 24)
    for target in ("joint", "blind", "marginal"):
        values = data.target(target)
        assert values.shape == (48, 9)
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-12)
    for target in ("composite", "composite_blind", "factor_m", "factor_phi"):
        values = data.target(target)
        assert values.shape == (48, 3)
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-12)

    # The action-conditioned filter must reproduce the environment exactly,
    # and the action-blind filter must not.
    np.testing.assert_allclose(data.joint, data.diagnostic, atol=1e-12)
    assert np.abs(data.blind - data.joint).max() > 1e-3

    # A perfect register report keeps the joint belief on the product manifold.
    assert data.product_state_gap()["joint_product_mse"] < 1e-20


def test_noisy_register_pushes_the_joint_belief_off_the_product_manifold():
    data = _probe("factoring_impossible", n_steps=64)
    assert data.marginal is None
    np.testing.assert_allclose(data.joint, data.diagnostic, atol=1e-12)
    assert data.product_state_gap()["joint_product_mse"] > 1e-4


def _oracle_probe_data(
    activations: np.ndarray,
    joint: np.ndarray,
    blind: np.ndarray,
) -> FeedbackProbeData:
    count = len(joint)
    zeros = np.zeros(count, dtype=np.int64)
    return FeedbackProbeData(
        activations=activations,
        joint=joint,
        diagnostic=joint,
        blind=blind,
        marginal=None,
        composite=np.zeros((count, 3)),
        composite_blind=np.zeros((count, 3)),
        factor_m=np.zeros((count, 3)),
        factor_phi=np.zeros((count, 3)),
        tokens=zeros,
        scored_tokens=zeros,
        previous_scored_tokens=zeros,
        actions=zeros,
        previous_actions=zeros,
        states=zeros,
        env_indices=zeros,
        episode_steps=np.arange(count, dtype=np.int64),
        rewards=np.zeros(count),
    )


def test_action_awareness_ratio_separates_aware_from_blind_features():
    """A representation carrying one belief must not decode the other."""

    process = composition.composed_process(OPERATING_STRENGTH, 1.0)
    rollout = composition.simulate_closed_loop(
        OPERATING_STRENGTH,
        1.0,
        n_chains=32,
        n_steps=288,
        burn_in=32,
        seed=21,
    )
    joint = rollout.beliefs.reshape(-1, 9)
    blind = composition.hmm_filter(
        rollout.tokens,
        process.transitions.mean(axis=0),
        process.emission,
    ).reshape(-1, 9)
    assert np.square(joint - blind).mean() > 1e-3
    half = len(joint) // 2

    ratios = {}
    for name, features in (("aware", joint), ("blind", blind)):
        train = _oracle_probe_data(features[:half], joint[:half], blind[:half])
        test = _oracle_probe_data(features[half:], joint[half:], blind[half:])
        fitted = {
            target: _fit_target(train, test, target)[0]
            for target in ("joint", "blind")
        }
        ratios[name] = (
            fitted["joint"]["global_mse_ratio"]
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
    design = {
        condition.name: (
            condition.feedback_strength,
            condition.register_noise,
            condition.observe_previous_guess,
        )
        for condition in CONDITIONS
    }
    assert design == {
        "factoring_free": (0.7, 0.0, True),
        "factoring_cheap": (0.7, 0.3, True),
        "factoring_costly": (0.7, 0.85, True),
        "factoring_impossible": (0.7, 1.0, True),
        "no_feedback": (0.0, 1.0, True),
        "deterministic_feedback": (1.0, 1.0, True),
        "factoring_free_blind": (0.7, 0.0, False),
        "factoring_impossible_blind": (0.7, 1.0, False),
    }
    assert TOTAL_ENV_STEPS == 2_500_000
    assert BASE_MODEL_CONFIG["context_length"] == CONTEXT_LENGTH
    assert OPERATING_STRENGTH == 0.7

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
        assert environment["model"]["kwargs"]["register_noise"] == (
            condition.register_noise
        )
        assert environment["task"]["kwargs"]["feedback_strength"] == (
            condition.feedback_strength
        )
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
    config = build_config(context, "factoring_costly")
    assert config.num_gpus_per_learner == 1
    assert config.num_gpus_per_env_runner == 0
