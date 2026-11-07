"""Architecture-only controls preserve the source recipes and probe contracts."""

import importlib
import json
from functools import partial
from types import FunctionType

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from envs.hmm import HMMEnv
from experiments.two_factor_reward_state_REINFORCE_cycle_4 import shared
from experiments.policy_baselines_2026_11 import (
    ReinforceGRU, ReinforceResidualMLP, recipe_overrides,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, partial):
        return (value.func, canonical(value.args), canonical(value.keywords))
    if isinstance(value, FunctionType):
        return (value.__module__, value.__qualname__,
                canonical(tuple(cell.cell_contents for cell in (value.__closure__ or ()))))
    return value


CASES = [
    (ladder, arch, condition)
    for ladder in (True, False)
    for arch in ("memoryless", "gru", "gru92")
    for condition in ((1, 2, 3) if ladder else ("reward_both", "reward_factor_1"))
]


@pytest.mark.parametrize("ladder,architecture,condition", CASES)
@pytest.mark.parametrize("smoke", [True, False])
def test_architecture_only_recipe_delta(tmp_path, ladder, architecture, condition, smoke):
    context = RunContext(experiment_dir=tmp_path, results_dir=tmp_path / "results",
                         artifacts_dir=tmp_path / "artifacts", seed=42, smoke=smoke,
                         hardware=PROFILES["cpu"])
    study = f"{'belief_quotient_ladder' if ladder else 'reward_relevance'}_{architecture}_2026_11"
    leaf = f"variant_{condition}" if ladder else condition
    new = importlib.import_module(f"experiments.{study}.{leaf}.experiment")
    source_path = (
        f"mess3_reward_state_action_symmetry_cycle_7.variant_{condition}_ctx32_ent"
        if ladder else f"two_factor_reward_state_REINFORCE_cycle_4.{condition}"
    )
    source = importlib.import_module(f"experiments.{source_path}.experiment")
    config, baseline = new.build_config(context), source.build_config(context)
    config.validate()
    baseline.validate()
    new_dict, old_dict = config.to_dict(), baseline.to_dict()
    for key in ("_rl_module_spec",):
        new_dict.pop(key, None)
        old_dict.pop(key, None)
    assert canonical(new_dict) == canonical(old_dict)
    spec = config.get_rl_module_spec()
    assert spec.module_class is (ReinforceGRU if architecture in ("gru", "gru92") else ReinforceResidualMLP)
    if architecture in ("gru", "gru92"):
        hidden_dim, n_layers = (92, 4) if architecture == "gru92" else (128, 2)
        assert spec.model_config == {"vf_share_layers": False, "hidden_dim": hidden_dim, "n_layers": n_layers, "max_seq_len": 32,
                                     "obs_feature_indices": None, "sampling_temperature": 1.0}
    else:
        assert spec.model_config == {"vf_share_layers": False, "d_model": 64, "n_layers": 6,
                                     "obs_feature_indices": tuple(range(3 if ladder else 9)),
                                     "sampling_temperature": 1.0}
    env = HMMEnv(config.env_config)
    module = spec.module_class(observation_space=env.observation_space,
                              action_space=env.action_space, model_config=spec.model_config)
    source_spec = baseline.get_rl_module_spec()
    transformer = source_spec.module_class(observation_space=env.observation_space,
                                           action_space=env.action_space,
                                           model_config=source_spec.model_config)
    count = sum(p.numel() for p in module.parameters())
    reference_count = sum(p.numel() for p in transformer.parameters())
    assert abs(count / reference_count - 1) < (0.04 if architecture == "gru92" else 0.02)
    if architecture == "gru92":
        assert count == (206_544 if ladder else 208_206)
    recipe = recipe_overrides(context, condition, study=study, architecture=architecture, ladder=ladder)
    assert recipe["parameter_count"] == count
    assert recipe["study"] == study
    assert recipe["model_config"] == spec.model_config
    assert recipe["architecture"] == architecture
    if ladder:
        assert recipe["entropy_coeff_schedule"] == source.entropy_schedule(context)
    env.close()


@pytest.mark.parametrize("cls", [ReinforceGRU, ReinforceResidualMLP])
def test_zero_baseline_temperature_and_probe_api(cls):
    env = HMMEnv(importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_7.shared").environment_config(2))
    module = cls(observation_space=env.observation_space, action_space=env.action_space,
                 model_config={"sampling_temperature": 2.0})
    obs = torch.randn(2, 6)
    state = {key: torch.from_numpy(value).unsqueeze(0).repeat(2, *([1] * value.ndim))
             for key, value in module.get_initial_state().items()}
    embedding, _ = module.encode_step(obs, state)
    residual, _ = module.encode_step_pre_final_norm(obs, state)
    torch.testing.assert_close(module.encoder.final_norm(residual), embedding)
    expected = module.heads.action_distribution_inputs(embedding) / 2
    for training in (True, False):
        torch.testing.assert_close(module._outputs(embedding, None, training=training)[Columns.ACTION_DIST_INPUTS], expected)
    for reference in (obs, embedding):
        values = module.compute_values({Columns.OBS: reference})
        assert values.dtype == reference.dtype and values.device == reference.device
        assert not torch.count_nonzero(values)
    env.close()


def test_cycle4_default_recipe_unchanged_and_empty_overrides(tmp_path):
    context = RunContext(experiment_dir=tmp_path, results_dir=tmp_path / "results",
                         artifacts_dir=tmp_path / "artifacts", seed=42, smoke=True,
                         hardware=PROFILES["cpu"])
    expected = shared._resolved_recipe(context, "reward_both")
    class StopAfterRecipe(Exception):
        pass
    def stop(*args, **kwargs):
        raise StopAfterRecipe
    with pytest.raises(StopAfterRecipe):
        shared.run_condition(context, "reward_both", config_builder=stop, recipe_overrides={})
    assert json.loads((context.results_dir / "resolved_recipe.json").read_text()) == expected
