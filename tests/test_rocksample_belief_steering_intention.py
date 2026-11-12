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


def test_report_rank_correlation_and_onset_pooling():
    from experiments.rocksample_jax_2026_10 import belief_steering_intention_report as report

    assert report.spearman([1, 2, 3, 4], [10, 20, 25, 90]) == 1.0
    assert report.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert report.ranks(np.array([5., 1., 5.])).tolist() == [1.5, 0., 1.5]
    cell = dict(eligible=10, sampled=4, intact_sampled=10, diverged=8, diverge_delay=2.0,
                diverge_at_onset=0.5, sample_delay=1.0)
    empty = dict(eligible=0, sampled=0, intact_sampled=0, diverged=0, diverge_delay=None,
                 diverge_at_onset=None, sample_delay=None)
    reports = [dict(cells={f"{rock}|Good→Bad": dict(relative=dict(steer={"2": cell if rock else empty}))
                           for rock in range(7)})]
    out = report.pooled_outcome(reports, "Good→Bad", "relative", "steer", 2)
    assert (out["eligible"], out["sampled"], out["diverged"]) == (60, 24, 48)
    assert out["diverge_delay"] == 2.0 and out["at_onset"] == 0.5
