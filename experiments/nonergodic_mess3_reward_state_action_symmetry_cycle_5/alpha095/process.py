"""Alternative dual MESS3 component parameters for the action-symmetry study.

Both components share alpha=0.95 (versus 0.60/0.66 in the default set);
mess3_a uses x=0.20 and mess3_b uses x=0.4666666666666667.
"""

from __future__ import annotations


COMPONENT_PARAMETERS = (
    {"name": "mess3_a", "x": 0.20, "alpha": 0.95},
    {"name": "mess3_b", "x": 0.4666666666666667, "alpha": 0.95},
)
