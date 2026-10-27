from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from envs.hmm import HMMEnv, TransitionEvent
from harness.cli import load_experiment, make_run_context

from experiments.pusher_b.process import EPISODE_LENGTH, pusher_b_model_b10
from experiments.pusher_b10_guess_or_pass_2026_10.process import (
    ppo_environment_config,
    sac_environment_config,
)
from experiments.pusher_b10_guess_or_pass_2026_10.campaign import (
    CAMPAIGN_RUNS,
    LEAF_COSTS,
    SOURCES,
)
from experiments.pusher_b10_guess_or_pass_2026_10.task import GuessOrPassTask
from experiments.pusher_b10_guess_or_pass_2026_10.transfer import (
    SOURCE_CATALOG,
    freeze_module_encoders,
    initialize_algorithm_transfer,
)


@pytest.mark.parametrize(
    ("action", "expected_reward", "metric"),
    ((0, 1.0, "guess_correct"), (1, -2.0, "guess_wrong"), (2, 0.0, "passed")),
)
def test_task_reward_components(action, expected_reward, metric):
    model = pusher_b_model_b10()
    task = GuessOrPassTask(model=model, cost=2.0)
    decision = task.resolve_action(action, 0, model)
    event = TransitionEvent(0, 0, 0, 0, 1)
    reward, components = task.reward(event, decision)
    assert reward == expected_reward
    assert components[metric] == 1.0
    assert set(components) == {
        "guess_correct",
        "guess_wrong",
        "passed",
        "bayes_expected_reward",
    }


@pytest.mark.parametrize(
    ("action", "exception"),
    ((-1, ValueError), (3, ValueError), (1.5, TypeError), (True, TypeError)),
)
def test_task_rejects_invalid_actions(action, exception):
    model = pusher_b_model_b10()
    task = GuessOrPassTask(model=model, cost=1.0)
    with pytest.raises(exception):
        task.resolve_action(action, 0, model)


@pytest.mark.parametrize(("cost", "expected"), ((1.0, 0.276), (2.0, 0.082)))
def test_bayes_reference_reward(cost, expected):
    environment = HMMEnv(ppo_environment_config(cost))
    totals = []
    for episode in range(2_000):
        environment.reset(seed=episode + 42)
        total = 0.0
        for _ in range(EPISODE_LENGTH):
            _, _, terminated, truncated, info = environment.step(2)
            total += info["reward_components"]["bayes_expected_reward"]
            if terminated or truncated:
                break
        totals.append(total / EPISODE_LENGTH)
    assert abs(float(np.mean(totals)) - expected) <= 0.01


@pytest.mark.parametrize(
    ("leaf", "cost", "algorithm"),
    (
        ("ppo_c1", 1.0, "ppo"),
        ("ppo_c1p5", 1.5, "ppo"),
        ("ppo_c2", 2.0, "ppo"),
        ("sac_c1", 1.0, "sac"),
        ("sac_c1p5", 1.5, "sac"),
        ("sac_c2", 2.0, "sac"),
    ),
)
def test_leaf_smoke_configs_use_three_actions(leaf, cost, algorithm):
    module = f"experiments.pusher_b10_guess_or_pass_2026_10.{leaf}.experiment"
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=42,
        smoke=True,
        hardware_profile="cpu",
    )
    if algorithm == "ppo":
        from experiments.pusher_b10_guess_or_pass_2026_10.ppo import build_config

        config = build_config(context, cost=cost)
        environment_config = ppo_environment_config(cost)
    else:
        from experiments.pusher_b10_guess_or_pass_2026_10.sac import build_config

        config = build_config(context, cost=cost)
        environment_config = sac_environment_config(cost)
    assert config.env_config["task"]["kwargs"]["cost"] == cost
    assert HMMEnv(environment_config).action_space.n == 3


class _FakeModule:
    def __init__(self, state):
        self.state = state

    def get_state(self):
        return {key: _copy_state_value(value) for key, value in self.state.items()}

    def set_state(self, state):
        self.state = {
            key: _copy_state_value(value) for key, value in state.items()
        }

    def make_target_networks(self):
        pass


class _ModuleBackedFakeModule:
    def __init__(self, module):
        self.module = module

    def get_state(self):
        return self.module.get_state()

    def set_state(self, state):
        self.module.set_state(state)

    def make_target_networks(self):
        pass


class _FakeLearnerGroup:
    def __init__(self, module):
        self.module = module

    def get_weights(self):
        return {"default_policy": self.module.get_state()}

    def set_weights(self, weights):
        self.module.set_state(weights["default_policy"])

    def foreach_learner(self, callback):
        callback(SimpleNamespace(module={"default_policy": self.module}))


class _FakeRunnerGroup:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def sync_weights(self, **kwargs):
        del kwargs
        self.algorithm.runner.set_state(
            self.algorithm.learner_group.get_weights()["default_policy"]
        )


class _FakeAlgorithm:
    def __init__(self, module):
        self.learner_group = _FakeLearnerGroup(module)
        self.runner = _FakeModule(module.get_state())
        self.env_runner_group = _FakeRunnerGroup(self)

    def get_module(self):
        return self.runner


def _copy_state_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return np.array(value, copy=True)


@pytest.mark.parametrize(
    ("algorithm", "keys"),
    (
        ("ppo", ("encoder.weight",)),
        (
            "sac",
            ("pi_encoder.weight", "qf_encoder.weight", "qf_twin_encoder.weight"),
        ),
    ),
)
def test_module_transfer_copies_encoders_and_keeps_fresh_heads(
    monkeypatch,
    tmp_path,
    algorithm,
    keys,
):
    from experiments.pusher_b10_guess_or_pass_2026_10 import transfer

    source_state = {key: np.ones((2, 2), dtype=np.float32) for key in keys}
    source_state["policy_head.weight"] = np.full((2, 2), 5.0, dtype=np.float32)
    target_state = {key: np.zeros((2, 2), dtype=np.float32) for key in keys}
    target_state["policy_head.weight"] = np.full((3, 2), 7.0, dtype=np.float32)
    source = _FakeModule(source_state)
    target = _FakeModule(target_state)
    monkeypatch.setattr(transfer.RLModule, "from_checkpoint", lambda _: source)
    recipe_path = tmp_path / "resolved_recipe.json"
    recipe_path.write_text("{}\n")
    report = initialize_algorithm_transfer(
        _FakeAlgorithm(target),
        module_checkpoint=tmp_path,
        algorithm_name=algorithm,
        trunk_mode="full",
        metadata={"source": "test"},
        recipe_path=recipe_path,
    )
    assert report["encoder_weights_copied"]
    for key in keys:
        np.testing.assert_array_equal(target.state[key], source_state[key])
    assert target.state["policy_head.weight"].shape == (3, 2)
    assert {
        "ppo_iter4",
        "ppo_iter143",
        "ppo_iter8",
        "ppo_ce_iter8",
        "sac_split_iter64",
    } <= set(SOURCE_CATALOG)


def test_ppo_ce_campaign_has_expected_unique_grid():
    assert LEAF_COSTS == (
        ("ppo_c2", 2.0),
        ("ppo_c1p5", 1.5),
        ("ppo_c1", 1.0),
    )
    assert SOURCES == (
        "ppo_iter4",
        "ppo_iter143",
        "ppo_ce_iter4",
        "ppo_ce_iter211",
        "ppo_iter8",
        "ppo_ce_iter8",
    )
    assert SOURCE_CATALOG["ppo_iter8"]["env_steps"] == 2_162_048
    assert SOURCE_CATALOG["ppo_iter8"]["pretraining_objective"] == "ppo"
    assert SOURCE_CATALOG["ppo_ce_iter8"] == {
        "algorithm": "ppo",
        "architecture": None,
        "leaf": "experiments/pusher_b/rl_b10_aux_ce",
        "run_id": "20260915T033547Z-913cef7f",
        "env_steps": 2_275_840,
        "probe_1_minus_r2": 0.1098,
        "greedy_accuracy": 0.6375,
        "training_iteration": 8,
        "checkpoint_index": "log_spaced_checkpoints/index.json",
        "pretraining_objective": "ppo+ntp_ce",
    }
    expected = [
        (leaf, None, None, cost, 42)
        for leaf, cost in LEAF_COSTS
    ]
    expected.extend(
        (leaf, source, "full", cost, 42)
        for source in SOURCES
        for leaf, cost in LEAF_COSTS
    )
    expected.extend(
        (leaf, source, "frozen", cost, 42)
        for source in SOURCES
        for leaf, cost in (("ppo_c2", 2.0),)
    )
    actual = {
        (run.leaf, run.source, run.trunk, run.cost, run.seed)
        for run in CAMPAIGN_RUNS
    }
    actual_order = [
        (run.leaf, run.source, run.trunk, run.cost, run.seed)
        for run in CAMPAIGN_RUNS
    ]
    assert len(CAMPAIGN_RUNS) == 27
    assert len({run.run_name for run in CAMPAIGN_RUNS}) == 27
    assert actual_order == expected
    assert actual == set(expected)
    assert all(run.run_name for run in CAMPAIGN_RUNS)
    from experiments.pusher_b10_guess_or_pass_2026_10.ppo import TOTAL_ENV_STEPS

    assert TOTAL_ENV_STEPS == 10_000_000


def test_auxiliary_ppo_source_transfers_only_encoder(monkeypatch, tmp_path):
    from ray.rllib.core.rl_module.rl_module import RLModule

    from experiments.pusher_b import rl as pusher_b_ppo
    from experiments.pusher_b.rl_b10_aux_ce.experiment import SETTINGS as CE_SETTINGS
    from experiments.pusher_b10_guess_or_pass_2026_10.ppo import (
        build_config as build_guess_or_pass_ppo_config,
        resolved_recipe,
    )

    experiment = load_experiment(
        "experiments.pusher_b10_guess_or_pass_2026_10.ppo_c2.experiment"
    )
    context = make_run_context(
        experiment,
        seed=42,
        smoke=True,
        hardware_profile="cpu",
    )
    source_config = pusher_b_ppo.build_config(
        context,
        preset="b10",
        settings=CE_SETTINGS,
    )
    target_config = build_guess_or_pass_ppo_config(context, cost=2.0)

    def build_module(config):
        environment = HMMEnv(config.env_config)
        spec = config.rl_module_spec
        spec.observation_space = environment.observation_space
        spec.action_space = environment.action_space
        return spec.build()

    source_module = build_module(source_config)
    target_module = build_module(target_config)
    source_state = source_module.get_state()
    target_state = target_module.get_state()
    aux_keys = {
        key
        for key in source_state
        if "aux" in key.lower() or "next_token" in key.lower()
    }
    assert aux_keys
    assert not aux_keys.intersection(target_state)
    assert HMMEnv(target_config.env_config).action_space.n == 3

    source_only_head_shape_differs = any(
        key in source_state
        and np.asarray(source_state[key]).shape != np.asarray(value).shape
        for key, value in target_state.items()
        if not key.startswith("encoder.")
    )
    assert source_only_head_shape_differs
    target_before = {
        key: np.asarray(value).copy()
        for key, value in target_state.items()
        if not key.startswith("encoder.")
    }

    monkeypatch.setattr(RLModule, "from_checkpoint", lambda _: source_module)
    recipe_path = tmp_path / "resolved_recipe.json"
    recipe_path.write_text("{}\n")
    algorithm = _FakeAlgorithm(_ModuleBackedFakeModule(target_module))
    report = initialize_algorithm_transfer(
        algorithm,
        module_checkpoint=tmp_path,
        algorithm_name="ppo",
        trunk_mode="full",
        metadata={
            "source": "ppo_ce_iter4",
            "pretraining_objective": "ppo+ntp_ce",
        },
        recipe_path=recipe_path,
    )

    transferred_state = target_module.get_state()
    for key in report["encoder_weight_keys"]:
        np.testing.assert_array_equal(
            np.asarray(transferred_state[key]),
            np.asarray(source_state[key]),
        )
    for key, value in target_before.items():
        np.testing.assert_array_equal(np.asarray(transferred_state[key]), value)
    assert report["encoder_weights_copied"]
    recipe = resolved_recipe(
        context,
        cost=2.0,
        transfer={"pretraining_objective": "ppo+ntp_ce"},
    )
    assert recipe["pretraining_objective"] == "ppo+ntp_ce"
    assert recipe["next_token_aux_coefficient"] == 0.0
    assert SOURCE_CATALOG["ppo_iter4"]["pretraining_objective"] == "ppo"
    assert SOURCE_CATALOG["ppo_ce_iter4"]["pretraining_objective"] == "ppo+ntp_ce"


@pytest.mark.parametrize("algorithm", ("ppo", "sac"))
def test_frozen_encoder_stays_fixed_while_heads_update(algorithm):
    module = torch.nn.Module()
    if algorithm == "ppo":
        module.encoder = torch.nn.Linear(2, 4)
        module.policy_head = torch.nn.Linear(4, 3)
        module.value_head = torch.nn.Linear(4, 1)
        encoder_names = ("encoder",)
        heads = (module.policy_head, module.value_head)

        def forward():
            return (
                module.policy_head(module.encoder(torch.ones(8, 2))).sum()
                + module.value_head(module.encoder(torch.ones(8, 2))).sum()
            )

    else:
        module.pi_encoder = torch.nn.Linear(2, 4)
        module.qf_encoder = torch.nn.Linear(2, 4)
        module.qf_twin_encoder = torch.nn.Linear(2, 4)
        module.pi = torch.nn.Linear(4, 3)
        module.qf = torch.nn.Linear(4, 3)
        module.qf_twin = torch.nn.Linear(4, 3)
        encoder_names = ("pi_encoder", "qf_encoder", "qf_twin_encoder")
        heads = (module.pi, module.qf, module.qf_twin)

        def forward():
            return sum(
                head(encoder(torch.ones(8, 2))).sum()
                for encoder, head in zip(
                    (
                        module.pi_encoder,
                        module.qf_encoder,
                        module.qf_twin_encoder,
                    ),
                    heads,
                )
            )

    freeze_module_encoders(module, algorithm)
    encoders = [getattr(module, name) for name in encoder_names]
    before_encoders = [
        parameter.detach().clone()
        for encoder in encoders
        for parameter in encoder.parameters()
    ]
    before_heads = [
        parameter.detach().clone()
        for head in heads
        for parameter in head.parameters()
    ]
    optimizer = torch.optim.Adam(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer.zero_grad()
    forward().backward()
    optimizer.step()
    after_encoders = [
        parameter.detach()
        for encoder in encoders
        for parameter in encoder.parameters()
    ]
    after_heads = [
        parameter.detach()
        for head in heads
        for parameter in head.parameters()
    ]
    assert all(torch.equal(before, after) for before, after in zip(before_encoders, after_encoders))
    assert any(not torch.equal(before, after) for before, after in zip(before_heads, after_heads))
