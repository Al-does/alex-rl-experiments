"""Actor-only checkpoint analysis for the discrete-SAC reproduction.

The SAC RLModule exposes the same explicit pre-final-LayerNorm actor hooks as
the PPO module. Reusing the battery keeps all sampling, regression, CEV, and
vary-one definitions identical while the SAC critics remain inaccessible to
the probes.
"""

from experiments.factored_representations_reproduction_PPO_2026_08.analysis import (
    analyze_checkpoint,
    cross_validated_svd_affine,
    plot_probe_trajectory,
)

__all__ = [
    "analyze_checkpoint",
    "cross_validated_svd_affine",
    "plot_probe_trajectory",
]
