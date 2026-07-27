"""Same head-to-head, at the points that keep the mixed-state gaps visible.

``operating_point_validation`` traded the Cantor structure away for global-R2
headroom. These two points keep it: ``cantor_sharp`` leaves ``alpha`` untouched
so the picture is exactly as crisp as the shipped one, and ``cantor`` takes the
widest band still showing visible level-one gaps.
"""

from __future__ import annotations

from experiments.mess3_token_guess_cycle_1.operating_point_validation.experiment import (
    run_points,
)
from experiments.mess3_token_guess_cycle_1.operating_points import FRACTAL_POINTS
from harness.context import RunContext


def run(context: RunContext):
    return run_points(context, FRACTAL_POINTS)
