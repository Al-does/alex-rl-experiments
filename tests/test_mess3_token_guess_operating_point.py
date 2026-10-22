"""Tests for choosing the MESS3 operating point."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.mess3_token_guess_cycle_2.operating_point import (
    PROBE_CHAINS,
    OperatingPoint,
    accuracy_bounds,
    belief_r2_floor,
    block_bootstrap_interval,
    context_requirement,
    integrated_autocorrelation,
    probe_steps_for_spread,
    receptive_field,
    replicate_probe_r2,
    simulate_parallel,
    smallest_sufficient_context_len,
)

CYCLE_1 = OperatingPoint(alpha=0.85, self_transition=0.90)
SLOW = OperatingPoint(alpha=0.70, self_transition=0.995)


def _streams(point, fit=20_000, test=10_000, seed=7):
    return (
        simulate_parallel(point, n_steps=fit, seed=seed),
        simulate_parallel(point, n_steps=test, seed=seed + 1),
    )


def test_operating_point_builds_valid_stochastic_matrices():
    for point in (CYCLE_1, SLOW):
        for matrix in (point.transition_matrix, point.emission_matrix):
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
            assert (matrix >= 0.0).all()
    assert CYCLE_1.transition_matrix[0, 0] == pytest.approx(0.90)
    assert CYCLE_1.emission_matrix[1, 1] == pytest.approx(0.85)


def test_slower_chains_persist_for_longer():
    assert SLOW.state_correlation_time > 10 * CYCLE_1.state_correlation_time
    assert CYCLE_1.state_correlation_time == pytest.approx(1 / (1 - 0.85), rel=1e-6)


def test_slowing_the_chain_lowers_the_no_network_floor():
    cycle_1_floor = belief_r2_floor(*_streams(CYCLE_1))
    slow_floor = belief_r2_floor(*_streams(SLOW))

    # Cycle 1's parameters leave the metric almost no room.
    assert cycle_1_floor > 0.95
    # The slower chain roughly triples what the metric can resolve.
    assert (1 - slow_floor) > 2.0 * (1 - cycle_1_floor)


def test_slowing_the_chain_also_widens_the_accuracy_range():
    _, cycle_1_test = _streams(CYCLE_1)
    _, slow_test = _streams(SLOW)
    cycle_1_low, cycle_1_high = accuracy_bounds(CYCLE_1, cycle_1_test)
    slow_low, slow_high = accuracy_bounds(SLOW, slow_test)

    assert cycle_1_high - cycle_1_low < 0.03
    assert slow_high - slow_low > 3.0 * (cycle_1_high - cycle_1_low)
    # The task has to stay learnable: well above chance.
    assert slow_high > 0.5


def test_a_64_observation_context_suffices_at_every_candidate():
    for point in (CYCLE_1, SLOW):
        _, test = _streams(point)
        scores = context_requirement(point, test, context_lengths=(8, 32, 64))
        assert scores[64] > 0.999, point
        assert scores[64] >= scores[32] >= scores[8]


def test_receptive_field_multiplies_the_band_by_the_layer_count():
    # CausalTransformerEncoder applies the band at every layer, so the study's
    # context_len of 64 over three layers reaches 192 observations.
    assert receptive_field(n_layers=3, context_len=64) == 192
    assert receptive_field(n_layers=3, context_len=16) == 48
    assert receptive_field(n_layers=1, context_len=16) == 16
    for bad in ({"n_layers": 0, "context_len": 8}, {"n_layers": 3, "context_len": 0}):
        with pytest.raises(ValueError):
            receptive_field(**bad)


def test_the_study_context_len_is_far_larger_than_the_belief_needs():
    for point in (CYCLE_1, SLOW):
        _, test = _streams(point)
        smallest = smallest_sufficient_context_len(point, test)
        assert smallest <= 16, point
        # Well inside the 192-observation reach the study currently pays for.
        assert receptive_field(n_layers=3, context_len=smallest) <= 48


def test_smallest_sufficient_context_len_needs_a_wide_enough_window():
    _, test = _streams(SLOW)
    narrow = simulate_parallel(SLOW, n_steps=2_000, seed=3, window=8)
    assert smallest_sufficient_context_len(SLOW, test) >= 4
    with pytest.raises(ValueError):
        smallest_sufficient_context_len(SLOW, narrow, candidates=(16, 20))


def test_replication_confirms_the_slow_chain_is_the_noisier_probe():
    fast = replicate_probe_r2(
        CYCLE_1, fit_steps=20_000, test_steps=10_000, replicates=8, seed=1
    )
    slow = replicate_probe_r2(
        SLOW, fit_steps=20_000, test_steps=10_000, replicates=8, seed=1
    )
    assert slow.std(ddof=1) > 3.0 * fast.std(ddof=1)


def test_probe_steps_for_spread_follows_the_square_root_law():
    # Measured by replication: +/-0.0025 at 30,000 steps on the slow chain.
    assert probe_steps_for_spread(
        0.0025, measured_steps=30_000, target_spread=0.0005
    ) == pytest.approx(750_000, rel=0.01)
    # Never asks for fewer steps than were already run.
    assert (
        probe_steps_for_spread(0.0005, measured_steps=30_000, target_spread=0.002)
        == 30_000
    )
    with pytest.raises(ValueError):
        probe_steps_for_spread(0.001, measured_steps=30_000, target_spread=0.0)


def test_slowing_the_chain_costs_independent_probe_samples():
    fast = integrated_autocorrelation(CYCLE_1, n_steps=60_000, seed=3)
    slow = integrated_autocorrelation(SLOW, n_steps=60_000, seed=3)

    assert fast < 30.0
    # The precision cost is what a wider metric is traded against.
    assert slow > 5.0 * fast


def test_a_realistically_collected_probe_has_a_much_wider_interval():
    # Sixteen long trajectories is how `collect_probe_data` gathers a rollout.
    # Four thousand short ones is nearly independent sampling, and flatters the
    # interval by hiding the correlation a real probe actually incurs.
    def interval(n_chains):
        fit = simulate_parallel(SLOW, n_steps=20_000, seed=7, n_chains=n_chains)
        test = simulate_parallel(SLOW, n_steps=10_000, seed=8, n_chains=n_chains)
        low, high = block_bootstrap_interval(
            fit, test, window=32, seed=5, resamples=120
        )
        return (high - low) / 2.0

    assert interval(PROBE_CHAINS) > 2.0 * interval(4_000)
    assert interval(PROBE_CHAINS) > 0.005
