import numpy as np

from experiments.two_factor_reward_state_REINFORCE_cycle_4.belief_intervention import (
    ACTION_PAIRS,
    PAIR_TO_ACTION,
    intervention_delta,
)


def test_factor_two_steering_preserves_factor_one_decoder_and_swaps_beliefs():
    rng = np.random.default_rng(5)
    decoder = rng.standard_normal((16, 4))
    inverse = np.linalg.pinv(decoder)
    bias = rng.normal(size=4)
    residual = rng.normal(size=(20, 16))
    mean = rng.normal(size=4)
    decoded = residual @ decoder + bias
    random_direction = rng.standard_normal(16)
    random_direction -= random_direction @ decoder @ inverse
    random_direction /= np.linalg.norm(random_direction)

    erased = intervention_delta(
        residual, decoder, bias, inverse, mean, "erase_2", random_direction
    )
    np.testing.assert_allclose((residual + erased) @ decoder + bias,
                               np.column_stack((decoded[:, :2],
                                                np.broadcast_to(mean[2:], (20, 2)))))

    swapped = intervention_delta(
        residual, decoder, bias, inverse, mean, "swap_2", random_direction
    )
    modified = (residual + swapped) @ decoder + bias
    np.testing.assert_allclose(modified[:, :2], decoded[:, :2], atol=1e-12)
    np.testing.assert_allclose(modified[:, 2], decoded[:, 3], atol=1e-12)
    np.testing.assert_allclose(modified[:, 3], decoded[:, 2], atol=1e-12)

    opposite = intervention_delta(
        residual, decoder, bias, inverse, mean, "opposite_2", random_direction
    )
    opposite_beliefs = (residual + opposite) @ decoder + bias
    expected = (decoded[:, 3] > decoded[:, 2]).astype(float)
    np.testing.assert_allclose(opposite_beliefs[:, 2], expected, atol=1e-12)
    np.testing.assert_allclose(opposite_beliefs[:, 3], 1 - expected, atol=1e-12)
    np.testing.assert_allclose(opposite_beliefs[:, :2], decoded[:, :2], atol=1e-12)

    control = intervention_delta(
        residual, decoder, bias, inverse, mean, "random_direction", random_direction
    )
    np.testing.assert_allclose(control @ decoder, 0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(control, axis=1),
                               np.linalg.norm(erased, axis=1), atol=1e-12)


def test_clamped_action_pairs_cover_independent_factor_choices():
    for intact in ACTION_PAIRS:
        for steered in ACTION_PAIRS:
            combined = ACTION_PAIRS[PAIR_TO_ACTION[(intact[0], steered[1])]]
            assert combined == (intact[0], steered[1])
