import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.two_factor_factor2_suppression_steering_eps02_jax_2026_11 import steering as s
from experiments.two_factor_factor2_suppression_steering_eps02_jax_2026_11.summarize import summarize


def test_targets_preserve_factor_one_and_suppress_factor_two():
    belief = np.array([.2, .3, .5])[:, None] * np.array([.6, .1, .3])[None, :]
    for target, decoded in (("factored", [[.2, .3, .6, .1]]), ("joint", [belief.ravel()[:8]])):
        out = s.suppression_target(jnp.asarray(decoded), target)
        factors = s.base.decoded_factor_beliefs(out, "marginal" if target == "factored" else target)
        np.testing.assert_allclose(factors[0, 0], [.2, .3, .5], atol=1e-6)
        np.testing.assert_allclose(factors[0, 1], [0, 1, 0], atol=1e-6)


def test_selection_and_empty_summary():
    sweep = {str(x): {"factor_reward": [0.5, .6 - .1 * x]} for x in s.base.STRENGTHS}
    assert s.select_strength(sweep, .5) == {"strength": 1., "reached_random_baseline": True}
    assert s.select_strength(sweep, 0.) == {"strength": 3., "reached_random_baseline": False}
    with pytest.raises(ValueError, match="no reports"):
        summarize([])


def test_summary_seed_standard_error_and_duplicate_rejection():
    reports = [{"source": {"seed": seed}, "confirmation": {"intact": {
        "factor_reward": [value, value], "total_reward": 2 * value}},
        "strength_tuning": {"selected": {t: {"reached_random_baseline": seed == 42}
                                         for t in s.TARGETS}}}
        for seed, value in ((42, .2), (43, .4))]
    result = summarize(reports)
    np.testing.assert_allclose(result["across_agent"]["intact"]["F1"]["mean"], .3)
    np.testing.assert_allclose(result["across_agent"]["intact"]["F1"]["seed_se"], .1)
    assert result["reached_random_baseline"] == {"factored": 1, "joint": 1}
    assert summarize(reports[:1])["across_agent"]["intact"]["F1"]["seed_se"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize([reports[0], reports[0]])


def test_zero_steering_null_norm_and_complete_episode_rewards():
    spec = s.base.models.ModelSpec(kind="transformer", obs_dim=18, n_actions=9, width=16,
                                   n_layers=1, context_len=4, chunk_len=4)
    params = s.base.models.init_params(spec, jax.random.PRNGKey(0))
    rng = np.random.default_rng(1)
    decoder = rng.normal(size=(16, 4))
    steer = {"decoder": jnp.asarray(decoder), "bias": jnp.zeros(4),
             "embedding": jnp.asarray(rng.normal(size=(4, 16))),
             "null_direction": jnp.asarray(s.base.null_direction(decoder, 2)),
             "strength": jnp.float32(.8), "use_null": jnp.bool_(False)}
    x = jnp.asarray(rng.normal(size=(3, 16)))
    shift = s.steering_shift(x, steer, "factored")
    null = s.steering_shift(x, {**steer, "use_null": jnp.bool_(True)}, "factored")
    np.testing.assert_allclose(np.linalg.norm(shift, axis=1), np.linalg.norm(null, axis=1), rtol=1e-5)
    np.testing.assert_allclose(null @ decoder, 0, atol=1e-5)
    process = s.source.source_process()
    kwargs = dict(episodes=2, seed=2042, site=0, target="factored")
    intact = s.evaluate(spec, params, process, **kwargs)
    zero = s.evaluate(spec, params, process, **kwargs, policy="steered",
                      steer={**steer, "strength": jnp.float32(0)})
    assert intact == zero
    assert intact["factor_action_change_fraction"] == [0, 0]
    assert intact["total_reward"] == sum(intact["factor_reward"])
    with pytest.raises(ValueError, match="at least two"):
        s.evaluate(spec, params, process, episodes=1, seed=0)


def test_fixed_joint_cli_skips_tuning_and_keeps_original_outputs(monkeypatch, tmp_path):
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    spec = s.base.models.ModelSpec(kind="transformer", obs_dim=18, n_actions=9, width=16,
                                   n_layers=4, context_len=4, chunk_len=4)
    process = s.source.source_process()
    (recipe_dir / "resolved_recipe.json").write_text(json.dumps({"model": asdict(spec),
        "process": {k: np.asarray(v).tolist() for k, v in process._asdict().items()}}))
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(s, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(s.source, "source_runs", lambda: [{"seed": 42, "results": recipe_dir}])
    monkeypatch.setattr(s, "fetch_checkpoint", lambda *args: (checkpoint, True))
    monkeypatch.setattr(s.base, "_git", lambda path: {"commit": "test", "dirty": False})
    monkeypatch.setattr(s.base, "load_params", lambda *args: None)
    rng = np.random.default_rng(3)
    rows = {"belief": rng.dirichlet(np.ones(9), 64), "sites": rng.normal(size=(64, 5, 16))}
    monkeypatch.setattr(s.base, "collect_rows", lambda *args, **kwargs: rows)
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["seed"] == 2042
        return {"factor_reward": [.4, .1], "total_reward": .5}

    monkeypatch.setattr(s, "evaluate", evaluate)
    monkeypatch.setattr(sys, "argv", ["steering", "--agent-seeds", "42", "--fixed-joint-strength", "1.0"])
    s.main()
    report_path = tmp_path / "results/joint_fixed_1.0/eps02_factor2_suppression__seed42.json"
    report = json.loads(report_path.read_text())
    assert report["fixed_joint_strength"] == 1.
    assert report["tune_episodes"] == 0 and "strength_tuning" not in report
    assert report["checkpoint_manifest_sha256_verified"]
    assert len(calls) == 8
    assert [c["site"] for c in calls[2:7]] == list(range(5))
    assert all(float(c["steer"]["strength"]) == 1 for c in calls[2:])
    assert bool(calls[-1]["steer"]["use_null"])
    with pytest.raises(FileExistsError):
        s.main()
    monkeypatch.setattr(sys, "argv", ["steering", "--fixed-joint-strength", "nan"])
    with pytest.raises(SystemExit):
        s.main()


@pytest.mark.parametrize("has_digest", [True, False])
def test_selectable_checkpoint_verification(monkeypatch, tmp_path, has_digest):
    payload = b"checkpoint"
    class Client:
        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(json.dumps({"files": [{
                "relative_path": "checkpoints/step_0000000123.npz",
                "sha256": hashlib.sha256(payload).hexdigest()}] if has_digest else []}).encode())}

        def download_file(self, bucket, key, path):
            Path(path).write_bytes(payload)

    class Config:
        bucket = "test"
        def s3_client(self):
            return Client()

    monkeypatch.setattr(s.source.B2StorageConfig, "from_env", lambda: Config())
    run = {"seed": 42, "run_id": "run", "remote_prefix": "prefix"}
    path, verified = s.fetch_checkpoint(run, tmp_path, 123)
    assert verified == has_digest and "step_0000000123" in path.name
    path.write_bytes(b"corrupt")
    if has_digest:
        with pytest.raises(ValueError, match="SHA-256"):
            s.fetch_checkpoint(run, tmp_path, 123)


def test_fixed_summary_ratios_and_count():
    reports = [{"source": {"seed": seed}, "fixed_joint_strength": 1., "checkpoint_step": 123,
        "confirmation": {c: {"factor_reward": pair, "total_reward": sum(pair)} for c, pair in
                         {"intact": [.4, .4], "random": [.1, .1],
                          "joint_block_4": [.3, f2]}.items()}} for seed, f2 in ((42, .05), (43, .15))]
    result = summarize(reports)
    assert result["joint_block_4_below_random"] == 1
    assert result["across_agent"]["joint_block_4"]["F2/random"] == pytest.approx({"mean": 1., "seed_se": .5})
    np.testing.assert_allclose(result["across_agent"]["joint_block_4"]["F1/intact"]["mean"], .75)
    reports[1]["fixed_joint_strength"] = .9
    with pytest.raises(ValueError, match="inconsistent fixed"):
        summarize(reports)
