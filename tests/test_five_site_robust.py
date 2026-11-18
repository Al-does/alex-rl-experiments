import math

import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import summarize_five_site_robust as robust


def complete_rows():
    return [dict(condition=name, step=step, site=site, target=target, seed=seed,
                 error=1000 if seed == 51 else seed - 42)
            for name in robust.stage2.NAMES.values() for step in robust.stage2.STEPS
            for site in robust.stage2.probe.SITES for target in robust.TARGETS
            for seed in robust.stage2.SEEDS]


def test_medians_keep_all_seeds_and_sensitivity_excludes_only_seed51():
    medians, sensitivity = robust.robust_statistics(complete_rows())
    assert len(medians) == 120
    assert set(medians.values()) == {7.0}
    assert len(sensitivity) == 2
    for row in sensitivity:
        assert row["mean"] == pytest.approx((sum(range(15)) - 9) / 14)
        assert row["n"] == 14 and 51 not in row["seeds"]


@pytest.mark.parametrize("mode", ["empty", "missing", "duplicate", "nonfinite"])
def test_incomplete_duplicate_and_nonfinite_grid_rejected(mode):
    rows = complete_rows()
    if mode == "empty":
        rows = []
    elif mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows.append(rows[0].copy())
    else:
        rows[0]["error"] = math.nan
    with pytest.raises(ValueError):
        robust.robust_statistics(rows)
