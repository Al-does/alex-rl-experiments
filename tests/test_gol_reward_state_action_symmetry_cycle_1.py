from __future__ import annotations

import importlib
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest

from harness.context import RunContext
from harness.hardware import PROFILES


STUDY = "experiments.gol_reward_state_action_symmetry_cycle_1"


@pytest.mark.parametrize("variant", (2, 3))
@pytest.mark.parametrize("speed", ("half", "quarter"))
def test_fresh_ppo_configs_and_observation_contract(tmp_path, variant, speed):
    from envs.hmm import HMMEnv

    suffix = "_quarter" if speed == "quarter" else ""
    leaf = importlib.import_module(f"{STUDY}.variant_{variant}{suffix}.experiment")
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    config = leaf.build_config(context)
    assert config is not leaf.build_config(context)
    assert config.gamma == 0.99
    assert config.lambda_ == 0.95
    assert config.batch_mode == "truncate_episodes"
    assert config.train_batch_size_per_learner == 1024
    assert config.minibatch_size == 128
    assert config.env_config["episode_length"] is None
    assert config.env_config["reset_emission"] is False
    assert config.env_config["task"]["kwargs"] == {"variant": variant, "speed": speed}
    assert config.env_config["model"]["kwargs"] == {"variant": variant, "speed": speed}
    assert config.num_env_runners == 0
    assert config.num_gpus_per_learner == 0
    json.dumps(config.env_config)
    env = HMMEnv(config.env_config)
    try:
        assert env.action_space.n == 4
        assert env.observation_space.shape == (6,)
        observation, info = env.reset(seed=42)
        np.testing.assert_array_equal(observation, np.zeros(6))
        assert "belief_current" not in info
        for action in range(4):
            observation, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(observation)
            assert observation[:2].sum() == 1
            np.testing.assert_array_equal(observation[2:], np.eye(4)[action])
            assert reward in (0.0, 1.0)
            assert not terminated and not truncated
    finally:
        env.close()
    full = leaf.build_config(replace(context, smoke=False))
    assert full.env_config == config.env_config
    assert full.rl_module_spec.model_config == config.rl_module_spec.model_config


@pytest.mark.parametrize("speed,occupancy", (("half", 0.4248120300751879), ("quarter", 0.3564668769716088)))
def test_design_matches_frozen_oracle_and_noninjectivity(speed, occupancy):
    from envs.gol.model import controlled_kernels
    from envs.hmm.model import stationary_distribution

    design = importlib.import_module(f"{STUDY}.design").analytic_design_summary(speed)
    for variant in (2, 3):
        report = design["variants"][f"variant_{variant}"]
        kernel = controlled_kernels(variant=variant, speed=speed)
        transitions = kernel.sum(axis=1)
        policies = list(itertools.product(range(4), repeat=4))
        rewards = [
            stationary_distribution(np.stack([transitions[a, s] for s, a in enumerate(policy)]))[2]
            for policy in policies
        ]
        expected = (0, 0 if variant == 2 else 1, 2, 3)
        assert policies[int(np.argmax(rewards))] == expected
        assert report["full_information_policy"] == list(expected)
        assert report["full_information_occupancy_upper_bound"] == pytest.approx(occupancy)
        assert report["reward_prediction_rank"] == (2 if variant == 2 else 3)
        assert report["marginal_null_error"] < 1e-14
    assert design["variant_2_exact_quotient_error"] < 1e-14
    assert design["variant_3_aggregation_error"] > 0.0
    json.dumps(design, allow_nan=False)


def test_two_variants_at_each_speed():
    from pathlib import Path

    root = Path(__file__).parents[1] / "experiments" / "gol_reward_state_action_symmetry_cycle_1"
    assert sorted(path.parent.name for path in root.glob("*/experiment.py")) == [
        "variant_2", "variant_2_quarter", "variant_3", "variant_3_quarter",
    ]


@pytest.mark.parametrize("speed", ("half", "quarter"))
def test_recipe_records_scientific_and_runtime_choices(tmp_path, speed):
    shared = importlib.import_module(f"{STUDY}.shared")
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=17,
        smoke=True,
    )
    recipe = shared.resolved_recipe(context, 2, speed=speed)
    assert recipe["seed"] == 17
    assert recipe["smoke"] is True
    assert recipe["total_env_steps"] == 2048
    assert recipe["previous_reward_in_observation"] is False
    assert recipe["filter_conditions_on_reward"] is False
    assert recipe["speed"] == speed
    assert recipe["analytic_design"]["speed"] == speed
    assert recipe["environment"]["model"]["kwargs"]["speed"] == speed
    assert recipe["variant"] == 2
    assert recipe["model"]["context_len"] == 64
    json.dumps(recipe, allow_nan=False)
    with pytest.raises(ValueError, match="continuation"):
        shared.run_condition(replace(context, resume_from=tmp_path / "checkpoint"), 2)


@pytest.mark.parametrize("variant", (2, 3))
@pytest.mark.parametrize("speed", ("half", "quarter"))
@pytest.mark.parametrize("smoke", (False, True))
def test_leaf_propagates_speed_and_budget_to_training_and_probes(tmp_path, monkeypatch, variant, speed, smoke):
    from types import SimpleNamespace

    shared = importlib.import_module(f"{STUDY}.shared")
    analysis = importlib.import_module(f"{STUDY}.analysis")
    suffix = "_quarter" if speed == "quarter" else ""
    leaf = importlib.import_module(f"{STUDY}.variant_{variant}{suffix}.experiment")
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=smoke,
    )
    expected_steps = 2048 if smoke else 2_500_000
    trained = []
    stops = []
    probes = []

    def train(config, context, **kwargs):
        trained.append(config.env_config)
        stops.append(kwargs["stop"])
        return [SimpleNamespace(error=None)]

    def probe(context, **kwargs):
        probes.append(kwargs)
        return {"speed": kwargs["speed"], "agent_steps": kwargs["agent_steps"]}

    monkeypatch.setattr(shared, "run_tune", train)
    monkeypatch.setattr(shared, "write_training_curves", lambda context: None)
    monkeypatch.setattr(shared, "checkpoint_records", lambda *args, **kwargs: [{
        "checkpoint_path": tmp_path / "final", "checkpoint_name": "final",
        "training_iteration": 2, "agent_steps": expected_steps,
    }])
    monkeypatch.setattr(analysis, "analyze_checkpoint", probe)
    summary = leaf.run(context)
    assert stops == [{"env_runners/num_env_steps_sampled_lifetime": expected_steps}]
    assert trained[0]["model"]["kwargs"] == {"variant": variant, "speed": speed}
    assert [(point["variant"], point["speed"], point["agent_steps"]) for point in probes] == [
        (variant, speed, 0), (variant, speed, expected_steps),
    ]
    assert summary["speed"] == speed
    recipe = json.loads((context.results_dir / "resolved_recipe.json").read_text())
    assert recipe["analytic_design"]["speed"] == speed
    assert recipe["total_env_steps"] == expected_steps
