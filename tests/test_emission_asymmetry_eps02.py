from dataclasses import asdict
import hashlib
import importlib
import json
from unittest.mock import patch

import jax
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.env import belief_reset, belief_update
from experiments.belief_quotient_history_jax_2026_11.training import _save
from experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11 import (
    analysis, design, environment as env, report, shared,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import (
    environment as source_env, shared as source,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo


def _history(seed=0, episodes=6, steps=40):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, (episodes, steps)), rng.integers(0, 3, (episodes, steps))


def test_zero_asymmetry_is_the_source_reward_state_process():
    for ours, theirs in zip(env.ladder_process(0.0), source_env.ladder_process(2)):
        np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("kappa", env.LEVELS.values())
def test_only_the_c_rate_of_states_1_and_2_changes(kappa):
    full, base = env.model(kappa), env.model(0.0)
    np.testing.assert_allclose(full["emission"].sum(1), 1)
    np.testing.assert_allclose(full["emission"][:, 2], [0.85 - kappa, 0.85 + kappa, 0.98])
    np.testing.assert_allclose(full["emission"][0, 0] / full["emission"][0, 1], 3)
    np.testing.assert_allclose(full["emission"][1, 1] / full["emission"][1, 0], 3)
    np.testing.assert_array_equal(full["transitions"], source_env.action_transitions(2))
    defect = env.lumpability_defect(full)
    assert defect["transition"] == 0 and defect["emission"] == pytest.approx(2 * kappa)
    for name, value in env.quotient_model(base).items():
        np.testing.assert_allclose(env.quotient_model(full)[name], value, atol=1e-15)


def test_asymmetry_bounds():
    for kappa in (-0.01, 0.13):
        with pytest.raises(ValueError):
            env.emission_matrix(kappa)


def test_full_filter_matches_the_training_filter():
    tokens, actions = _history()
    process = env.ladder_process(0.05)
    belief = env.filter_histories(env.model(0.05), tokens, actions)
    for episode in range(len(tokens)):
        expected = belief_reset(process, tokens[episode, 0])
        for t in range(tokens.shape[1]):
            if t:
                expected = belief_update(process, expected, actions[episode, t - 1],
                                         tokens[episode, t])
            np.testing.assert_allclose(belief[episode, t], expected, atol=2e-5)


def test_quotient_filter_is_exact_only_without_asymmetry():
    tokens, actions = _history(1, steps=200)
    merged = (tokens == 2).astype(int)

    def gap(kappa):
        full = env.model(kappa)
        coarse = {**full, "emission": np.stack(
            [full["emission"][:, :2].sum(1), full["emission"][:, 2]], 1)}
        exact = env.filter_histories(coarse, merged, actions)[..., 2]
        return np.abs(env.quotient_filter(full, tokens, actions) - exact).max()

    assert gap(0.0) < 1e-12
    assert gap(0.05) > 1e-2


@pytest.mark.parametrize("level", env.LEVELS)
def test_leaves_keep_the_source_recipe(level):
    module = importlib.import_module(f"experiments.{shared.STUDY}.asym_{level}_transformer.experiment")
    reference_module = importlib.import_module(
        f"experiments.{source.STUDY}.variant_2_transformer.experiment")
    for smoke in (False, True):
        context = type("Context", (), {"smoke": smoke})()
        with patch.object(shared, "run_study", side_effect=lambda *a, **kw: kw), \
                patch.object(source.base, "run_study", side_effect=lambda *a, **kw: kw):
            result, reference = module.run(context), reference_module.run(context)
        assert asdict(result["config"]) == asdict(reference["config"])
        assert result["spec"] == reference["spec"]
        assert result["representation"] == reference["representation"]
        assert (result["study"], result["condition"]) == (shared.STUDY, f"asym_{level}")
        for key in ("initial", "transitions", "reward", "reward_components"):
            np.testing.assert_array_equal(getattr(result["process"], key),
                                          getattr(reference["process"], key))
        kappa = env.LEVELS[level]
        np.testing.assert_allclose(result["process"].emission, env.emission_matrix(kappa))
        extra = json.loads(json.dumps(result["recipe_extra"]))
        assert extra["emission_asymmetry_kappa"] == kappa and extra["variant"] == 2
        assert extra["lumpability_defect"]["emission"] == pytest.approx(2 * kappa)
        targets = result["targets"](np.array([[0.5, 0.2, 0.3]]))
        np.testing.assert_allclose(targets["reward_class"], [[0.3]])
        np.testing.assert_allclose(targets["identity"], [[0.3]])


def test_reference_controllers():
    full = env.model(0.1)
    kwargs = {"episodes": 48, "seed": 3, "horizon": 96, "warmup": 32}
    first = design.simulate(full, "quotient", 0.18, **kwargs)
    again = design.simulate(full, "quotient", 0.18, **kwargs)
    np.testing.assert_array_equal(first["expected_reward"], again["expected_reward"])
    assert first["rows"]["rho"].shape == (48 * 64,)
    assert design.simulate(full, "quotient", 0.0, **kwargs)["tilt_x_fraction"] == 0
    assert design.simulate(full, "reward_marginal", 1.1, **kwargs)["tilt_x_fraction"] == 1
    with pytest.raises(ValueError):
        design.simulate(full, "identity", 0.18, **kwargs)
    result = design.reference_controllers(full, thresholds=(0.1, 0.2), **kwargs)
    for statistic in design.STATISTICS:
        assert result[statistic]["selected_threshold"] in (0.1, 0.2)
        assert 0 < result[statistic]["expected_reward"]["mean"] < 1
    assert result["confirmation_seed"] == 4
    assert 0 < result["target_separation"]["reward_marginal_unexplained_by_quotient"] < 1
    rows = {name: np.linspace(0, 1, 50) for name in ("rho", "quotient", "identity")}
    assert design.target_separation(rows)["reward_marginal_unexplained_by_quotient"] < 1e-12


def test_action_rule_lookup_recovers_a_threshold_rule():
    rng = np.random.default_rng(0)
    fit, test = rng.random((2, 4000))
    other = rng.random((2, 4000))
    fit_action, test_action = (fit < 0.3).astype(int), (test < 0.3).astype(int)
    agreement = (analysis.rule_predictions([fit], fit_action, [test], 3) == test_action).mean()
    assert agreement > 0.98
    blind = (analysis.rule_predictions([other[0]], fit_action, [other[1]], 3) == test_action).mean()
    assert blind == pytest.approx(0.7, abs=0.05)


def _write_run(tmp_path, *, horizon=192, final_step=4096, use_ppo=False, training_smoke=False):
    process = env.ladder_process(0.05)
    spec = models.architecture_spec("transformer", obs_dim=6, n_actions=3, n_tokens=3,
                                    context_len=32)
    run_dir, checkpoint_dir = tmp_path / "results" / "run", tmp_path / "checkpoints"
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    init_fn = ppo.init_params if use_ppo else models.init_params
    for step, seed in ((0, 0), (final_step, 1)):
        _save(checkpoint_dir / f"step_{step:010d}.npz",
              init_fn(spec, jax.random.PRNGKey(seed)))
    training = {"episode_length": horizon}
    if use_ppo:
        training["algorithm"] = "PPO"
    files = {
        "resolved_recipe": {
            "study": shared.STUDY, "condition": "asym_050", "seed": 7, "framework": "jax",
            "continued_from": None, "smoke": training_smoke, "model": asdict(spec),
            "training": training, "emission_asymmetry_kappa": 0.05,
            "process": {name: np.asarray(value).tolist()
                        for name, value in process._asdict().items()},
        },
        "run_manifest": {"status": "completed", "run_id": "run", "git": None},
        "condition_summary": {"training": {"env_steps": final_step}},
    }
    for name, content in files.items():
        (run_dir / f"{name}.json").write_text(json.dumps(content))
    return run_dir, checkpoint_dir


def test_checkpoints_are_verified_against_the_durability_manifest(tmp_path):
    run_dir, checkpoint_dir = _write_run(tmp_path)
    run = analysis.load_run(run_dir)
    with pytest.raises(ValueError, match="durability manifest"):
        analysis.verified_checkpoints(run, checkpoint_dir)
    records = [{"relative_path": f"checkpoints/{path.name}", "key": f"prefix/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size}
               for path in sorted(checkpoint_dir.glob("step_*.npz"))]
    manifest = checkpoint_dir / "durability_manifest.json"
    manifest.write_text(json.dumps({"prefix": "prefix", "bucket": "bucket", "files": records}))
    checked = analysis.verified_checkpoints(run, checkpoint_dir)
    assert checked["final"]["step"] == 4096
    assert all(record["verified_against_durability_manifest"] for record in checked.values())
    records[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"prefix": "prefix", "bucket": "bucket", "files": records}))
    with pytest.raises(ValueError, match="does not match"):
        analysis.verified_checkpoints(run, checkpoint_dir)


@pytest.mark.parametrize("use_ppo,training_smoke", [
    (False, False), (True, False), (True, True),
], ids=["reinforce", "ppo", "ppo-smoke-training"])
def test_analysis_and_report_end_to_end(tmp_path, monkeypatch, use_ppo, training_smoke):
    run_dir, checkpoint_dir = _write_run(tmp_path, use_ppo=use_ppo, training_smoke=training_smoke)
    monkeypatch.setattr(analysis, "BUDGET", analysis.SMOKE_BUDGET)
    result = analysis.analyze(run_dir, checkpoint_dir, smoke=not training_smoke,
                              allow_unverified=training_smoke)
    json.dumps(result)
    assert result["lumpability_defect"]["emission"] == pytest.approx(0.1, abs=1e-6)
    assert result["protocol"]["n_test_rows"] == 8 * (192 - analysis.WARMUP)
    assert result["protocol"]["filter_max_error"] < analysis.FILTER_TOLERANCE
    assert set(result["probes"]) == {"full_belief", "reward_marginal", "identity", "quotient",
                                     "quotient_occupancy_weighted", "quotient_gap"}
    assert set(result["probes"]["quotient"]["scores"]) == set(analysis.FEATURES)
    assert set(result["which_target"]["quotient"]) == {"trained", "initialization",
                                                       "training_change"}
    assert set(result["action_rules"]["agreement"]) == set(analysis.RULES)
    assert result["behavior"]["references"]["quotient_regret"]["mean"] is not None
    assert not result["checkpoints"]["final"]["verified_against_durability_manifest"]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="smoke"):
        report.summarize([path])
    path.write_text(json.dumps({**result, "smoke": False}))
    table = report.summarize([path])
    assert "asym_050" in table and len(table.splitlines()) == 3
