import numpy as np

from experiments.rocksample_jax_2026_10 import belief_steering_intention as intention


def test_next_sample_labels_the_upcoming_rock_and_its_lead():
    picked = np.zeros((1, 6, 3), bool)
    picked[0, 1, 2] = picked[0, 4, 0] = True
    rock, lead = intention.next_sample(picked)
    assert rock[0].tolist() == [2, 2, 0, 0, 0, 3]
    assert lead[0, :5].tolist() == [1, 0, 2, 1, 0]


def test_first_returns_axis_length_when_never_true():
    mask = np.array([[False, True, True], [False, False, False]])
    assert intention.first(mask).tolist() == [1, 3]


def test_linear_classifier_recovers_separable_labels():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 200)
    x = np.eye(3)[labels] @ rng.normal(size=(3, 5)) + 0.01 * rng.normal(size=(200, 5))
    model = intention.classifier(x, labels, 3)
    scores = intention.score(intention.predict(model, x), labels, 3)
    assert scores["accuracy"] == 1.0 and scores["macro_f1"] == 1.0


def test_onset_modes_use_the_frozen_protocol_and_twin_patches():
    assert intention.MODES["steer"] == intention.confirm.steer()
    assert intention.MODES["current_token"]["patch_sites"] == (intention.confirm.SITE,)
    assert intention.MODES["all_tokens"]["patch_sites"] == (intention.cf.SITES - 1,)
