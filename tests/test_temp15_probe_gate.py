import pytest

from experiments.within_class_belief_probes_2026_11.temp15 import gate_passes


@pytest.mark.parametrize("delta,ratio_delta,expected", [
    (0.0, 0.0, True),
    (1e-2, -1e-2, True),
    (3.797e-5, -3.797e-5, True),
    (2.185e-3, -2.185e-3, True),
    (0.0, 1.001e-2, False),
    (float("nan"), 0.0, False),
    (0.0, float("inf"), False),
])
def test_temp15_gate_checks_both_scores(delta, ratio_delta, expected):
    assert gate_passes({"delta": delta, "ratio_delta": ratio_delta}) is expected


def test_temp15_gate_uses_larger_five_se_threshold():
    assert gate_passes({"delta": .02, "ratio_delta": -.02, "bootstrap_se": .005})
    assert not gate_passes({"delta": .026, "ratio_delta": -.026, "bootstrap_se": .005})
