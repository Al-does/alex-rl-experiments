"""Unit tests for the asymmetric-chain and SAC Reward-State ablation studies."""
from __future__ import annotations

import math

import numpy as np

from experiments.belief_quotient_asymmetric_chain_eps02_jax_temp15_2026_11 import (
    shared as chain,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_ablation_2026_11 import (
    shared as sac_ablation,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11 import (
    shared as sac_autoalpha,
)


def test_asymmetric_chain_geometry():
    proc = chain.chain_process()
    assert proc.transitions.shape == (3, 3, 3)
    assert np.allclose(proc.transitions[0], proc.transitions[1])
    assert np.allclose(proc.transitions[0], proc.transitions[2])
    assert np.allclose(proc.transitions.sum(axis=-1), 1.0)
    assert np.isclose(np.linalg.norm(chain.U_V), 1.0)
    assert np.isclose(np.linalg.norm(chain.U_PERP), 1.0)
    assert np.isclose(chain.U_V @ np.ones(3), 0.0, atol=1e-12)
    assert np.isclose(chain.U_V @ chain.U_PERP, 0.0, atol=1e-12)
    # Angle between U_V and RHO_DIR should be ~30 degrees (clearly separated from rho).
    cos_angle = abs(float(chain.U_V @ chain.RHO_DIR))
    assert 0.80 < cos_angle < 0.90
    targets = chain.chain_targets(np.eye(3))
    assert set(targets) == {"belief", "rho", "delta", "y_v", "y_perp"}
    assert targets["y_v"].shape == (3, 1)


def test_sac_ablation_hyperparameters():
    a0 = sac_ablation.hyperparameters("alpha0", n_actions=3)
    assert a0["auto_entropy_tuning"] is False
    assert a0["alpha"] == 0.0
    assert a0["tau"] == 0.005
    assert a0["lr"] == sac_autoalpha.HYPERPARAMETERS["lr"]
    assert a0["critic_lr"] == sac_autoalpha.HYPERPARAMETERS["critic_lr"]
    assert a0["q_head_init"] == "zeros"

    t05 = sac_ablation.hyperparameters("tau05", n_actions=3)
    assert t05["auto_entropy_tuning"] is True
    assert t05["tau"] == 0.05
    assert math.isclose(t05["target_entropy"], 0.3 * math.log(3))
    assert t05["lr"] == sac_autoalpha.HYPERPARAMETERS["lr"]
    assert t05["critic_lr"] == sac_autoalpha.HYPERPARAMETERS["critic_lr"]
