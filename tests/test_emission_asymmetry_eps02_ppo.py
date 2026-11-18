import importlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11 import (
    environment,
    shared as baseline,
)
from experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_ppo_2026_11 import shared
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import (
    ppo,
    shared as ladder_ppo,
)


@pytest.mark.parametrize("level,kappa", [
    ("000", 0.0), ("025", 0.025), ("050", 0.05), ("075", 0.075), ("100", 0.10),
])
@pytest.mark.parametrize("smoke", [False, True])
def test_leaf_uses_original_process_with_exploratory_ppo(level, kappa, smoke):
    leaf = importlib.import_module(f"experiments.{shared.STUDY}.asym_{level}_transformer.experiment")
    ctx = SimpleNamespace(smoke=smoke, seed=0)
    with patch.object(shared, "run_study", side_effect=lambda context, **kw: (context, kw)):
        actual_ctx, actual = leaf.run(ctx)
    with patch.object(ladder_ppo.base, "run_study", side_effect=lambda context, **kw: kw):
        anchor = ladder_ppo.run_condition(ctx, variant=2)

    assert actual_ctx is ctx
    assert actual["study"] == shared.STUDY
    assert actual["condition"] == f"asym_{level}"
    assert actual["recipe_extra"]["emission_asymmetry_kappa"] == kappa
    assert actual["config"] == replace(anchor["config"], entropy_schedule=(
        (0, 0.03), (1_800_000_000, 0.0),
    ))
    assert actual["config"].total_env_steps == (4096 if smoke else 2_000_000_000)
    assert actual["config"].algorithm == "PPO"
    assert actual["trainer"] is ppo.train
    assert actual["spec"] == anchor["spec"]
    assert actual["representation"] == anchor["representation"]
    assert actual["targets"] is baseline.belief_targets

    for name, array in environment.ladder_process(kappa)._asdict().items():
        np.testing.assert_array_equal(getattr(actual["process"], name), array)
        if kappa == 0 or name != "emission":
            np.testing.assert_array_equal(array, getattr(anchor["process"], name))
