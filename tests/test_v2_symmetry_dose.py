import importlib

import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
from experiments.belief_quotient_v2_symmetry_dose_2026_11 import shared

LEAVES = {"lam_p100": (1., 0.), "lam_p075": (.75, 0.), "lam_p050": (.5, 0.), "lam_p000": (0., 0.),
          "lam_m050": (-.5, 0.), "lam_m100": (-1., 0.), "kappa_050": (1., .05), "kappa_100": (1., .1)}


def test_endpoints_are_the_v2_quotients():
    for lam, variant in ((1., 2), (-1., 3)):
        ours, theirs = shared.dose_process(lam, 0.), v2.ladder_process(variant)
        for a, b in zip(ours, theirs):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("lam,kappa", sorted(set(LEAVES.values())))
def test_kernels_are_stochastic_and_only_the_named_knob_moves(lam, kappa):
    transitions, emission = shared.action_transitions(lam), shared.emission_matrix(kappa)
    np.testing.assert_allclose(transitions.sum(-1), 1)
    np.testing.assert_allclose(emission.sum(-1), 1)
    reference = v2.action_transitions(2)
    np.testing.assert_array_equal(transitions[:, [0, 2]], reference[:, [0, 2]])
    np.testing.assert_array_equal(transitions[0], reference[0])
    np.testing.assert_allclose(emission[:, 2], [.85 - kappa, .85 + kappa, 1])
    np.testing.assert_allclose(emission[0, 0] / emission[0, 1], 3)
    np.testing.assert_allclose(emission[1, 1] / emission[1, 0], 3)


def test_rejects_out_of_range_knobs():
    with pytest.raises(ValueError):
        shared.action_transitions(1.5)
    with pytest.raises(ValueError):
        shared.emission_matrix(.15)


@pytest.mark.parametrize("leaf", sorted(LEAVES))
def test_leaf_passes_its_own_knobs(leaf, monkeypatch):
    module = importlib.import_module(f"experiments.belief_quotient_v2_symmetry_dose_2026_11.{leaf}.experiment")
    seen = {}
    monkeypatch.setattr(module, "run_condition", lambda context, **kwargs: seen.update(kwargs))
    module.run(None)
    assert seen == {"condition": leaf, "lam": LEAVES[leaf][0], "kappa": LEAVES[leaf][1]}
