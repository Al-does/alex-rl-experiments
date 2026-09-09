from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from envs.hmm import HMMEnv
from envs.strata.model import controlled_kernels, strata_model
from experiments.wing_two_factor_explore_cycle_1 import shared as cycle_1
from experiments.wing_two_factor_explore_cycle_1.model import WingActorCritic
from experiments.strata_two_factor_explore_cycle_2 import analysis, shared
from experiments.strata_two_factor_explore_cycle_2.process import (
    CONDITIONS,
    CONTROL_STRENGTH,
    REWARD_STATE,
    environment_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES


@pytest.fixture
def context(tmp_path):
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        seed=42,
        hardware=PROFILES["cpu"],
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_recipe_preserves_cycle_2_training_choices(context, condition):
    leaf = importlib.import_module(
        f"experiments.strata_two_factor_explore_cycle_2.{condition}_state_0.experiment"
    )
    config = leaf.build_config(context)
    old = cycle_1.build_config(context, condition, 0)
    assert config is not leaf.build_config(context)
    assert REWARD_STATE == 0
    assert CONTROL_STRENGTH == 1.0
    assert config.vf_clip_param == shared.VALUE_CLIP_PARAM == 1e9
    assert old.vf_clip_param == 10.0
    assert old.env_config["task"]["kwargs"]["strength"] == 0.15
    assert config.env_config == environment_config(condition)
    assert config.env_config["model"]["kwargs"]["factors"][0] == {
        "factory": "envs.strata.model:strata_model",
        "kwargs": {"alpha": 0.98, "t0": 0.30, "t1": 0.80},
    }
    assert (
        config.env_config["task"]["class"]
        == "envs.strata.tasks.reward_state:StrataRewardTask"
    )
    assert config.rl_module_spec.model_config == old.rl_module_spec.model_config
    for name in ("lr", "gamma", "lambda_", "clip_param", "grad_clip", "vf_loss_coeff", "entropy_coeff", "num_epochs"):
        assert getattr(config, name) == getattr(old, name)
    assert config.train_batch_size_per_learner == old.train_batch_size_per_learner == 1024
    assert config.minibatch_size == old.minibatch_size == 128
    report = shared.resolved_recipe(context, condition)
    assert report["study"] == "strata_two_factor_explore_cycle_2"
    assert report["value_clip_param"] == 1e9
    assert report["analytic_design"]["strength"] == 1.0
    assert report["analytic_design"]["alpha"] == 0.98
    assert report["analytic_design"]["t0"] == 0.30
    assert report["analytic_design"]["t1"] == 0.80
    assert (
        report["analytic_design"]["identifiability"][
            "max_one_step_null_residual"
        ]
        < 1e-12
    )
    assert report["total_env_steps"] == 2048
    full = shared.resolved_recipe(replace(context, smoke=False), condition)
    assert full["total_env_steps"] == (50_000_000 if condition == "reward_both" else 30_000_000)
    assert full["train_batch_size_per_learner"] == shared.TRAIN_BATCH_SIZE == 32_768
    assert full["minibatch_size"] == shared.MINIBATCH_SIZE == 4_096
    assert full["previous_reward_in_observation"] is False


def test_rotations_preserve_emission_map_and_exact_transducer_filter():
    config = environment_config("reward_both")
    config["diagnostics"] = {"belief": True, "tokens": True, "state": True, "transitions": True}
    config["randomize_first_episode_length"] = False
    env = HMMEnv(config)
    try:
        observation, info = env.reset(seed=13)
        belief = env.model.initial_distribution @ env.model.edge_transition_matrices[info["raw_token_current"]]
        belief /= belief.sum()
        assert observation.shape == (10,)
        assert observation[4:].sum() == 0
        np.testing.assert_allclose(info["belief_current"], belief)
        factor = controlled_kernels(strength=1.0)
        base = strata_model().edge_transition_matrices
        for action, shift in enumerate((0, 1, -1)):
            np.testing.assert_allclose(factor[action], np.roll(base, shift, axis=2))
        for action in range(9):
            a, b = divmod(action, 3)
            observation, reward, _, _, info = env.step(action)
            kernels = np.stack([np.kron(left, right) for left in factor[a] for right in factor[b]])
            np.testing.assert_allclose(info["executed_edge_transition_matrices"], kernels)
            np.testing.assert_allclose(kernels.sum(axis=2).T, env.model.emission_matrix)
            np.testing.assert_allclose(np.full(9, 1 / 9) @ kernels.sum(axis=0), np.full(9, 1 / 9))
            belief = belief @ kernels[info["raw_token_current"]]
            belief /= belief.sum()
            np.testing.assert_allclose(info["belief_current"], belief)
            states = divmod(info["state_current"], 3)
            assert reward == sum(state == 0 for state in states) / 2
            assert observation[4:].sum() == 2
    finally:
        env.close()


def test_ppo_value_errors_above_old_cap_still_receive_gradients(context):
    gradients = []
    for config in (cycle_1.build_config(context, "reward_both", 0), shared.build_config(context, "reward_both")):
        values = torch.zeros(2, requires_grad=True)
        module = SimpleNamespace(
            get_train_action_dist_cls=lambda: TorchCategorical,
            get_exploration_action_dist_cls=lambda: TorchCategorical,
            compute_values=lambda *args, **kwargs: values,
        )
        module.unwrapped = lambda: module
        learner = SimpleNamespace(
            module={"m": module},
            entropy_coeff_schedulers_per_module={"m": SimpleNamespace(get_current_value=lambda: 0.0)},
            metrics=SimpleNamespace(log_dict=lambda *args, **kwargs: None),
        )
        logits = torch.zeros(2, 3)
        batch = {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.ACTIONS: torch.zeros(2, dtype=torch.long),
            Columns.ACTION_LOGP: torch.full((2,), -np.log(3)),
            Postprocessing.ADVANTAGES: torch.zeros(2),
            Postprocessing.VALUE_TARGETS: torch.tensor([1.0, 5.0]),
        }
        loss = PPOTorchLearner.compute_loss_for_module(
            learner, module_id="m", config=config, batch=batch,
            fwd_out={Columns.ACTION_DIST_INPUTS: logits},
        )
        loss.backward()
        gradients.append(values.grad)
    torch.testing.assert_close(gradients[0], torch.tensor([-0.5, 0.0]))
    torch.testing.assert_close(gradients[1], torch.tensor([-0.5, -2.5]))


def test_checkpoint_probes_use_cycle_2_dynamics(context, monkeypatch):
    env = HMMEnv(environment_config("reward_factor_1"))
    module = WingActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_config=dict(cycle_1.MODEL_CONFIG),
    )
    env.close()
    monkeypatch.setattr(
        analysis,
        "load_algorithm",
        lambda _: nullcontext(SimpleNamespace(get_module=lambda: module)),
    )
    seen = []
    original = analysis.HMMEnv

    def collect_environment(config):
        seen.append(config["task"]["kwargs"]["strength"])
        return original(config)

    monkeypatch.setattr(analysis, "HMMEnv", collect_environment)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = analysis.analyze_checkpoint(
            context, checkpoint=context.artifacts_dir / "checkpoint",
            condition="reward_factor_1", checkpoint_label="initial",
            agent_steps=0, training_iteration=0,
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert seen and set(seen) == {1.0}
    assert report["reward_state"] == 0
    assert report["n_fit"] == report["n_test"] == 256
    assert report["product_consistency_max_abs"] < 1e-10
    assert set(report["layers"]) == {"layer_1", "layer_2", "layer_3"}
    assert (context.results_dir / "probe_battery.json").is_file()
