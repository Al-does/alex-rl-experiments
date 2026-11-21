import jax
import json
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_causal_decoding_2026_11 import decoders as d
from experiments.reward_factors_causal_decoding_2026_11.campaign_report import collect, mean_sd, SEEDS
from experiments.reward_factors_causal_decoding_2026_11.run import METHODS, TESTS


def test_campaign_report_requires_complete_confirmation_budget(tmp_path):
    assert mean_sd([1., 2., 3.]) == {"mean": 2., "sd": 1.}
    with pytest.raises(FileNotFoundError):
        collect(tmp_path)
    first = None
    for test in TESTS:
        study = "belief_quotient" if test.startswith("ladder") else "reward_factors"
        folder = tmp_path / f"experiments/{study}_causal_decoding_2026_11/results"
        folder.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            for method in METHODS:
                row = dict(test=test, seed=seed, method=method, campaign=True, smoke=False, pilot=False,
                           fit_episodes=24, tuning_episodes=16, confirmation_episodes=32, horizon=1024,
                           checkpoint={"sha256": f"{test}-{seed}"})
                path = folder / f"campaign_{test}_seed{seed}_{method}.json"
                path.write_text(json.dumps(row))
                if first is None:
                    first = path
    data, manifest = collect(tmp_path)
    assert len(data) == len(manifest) == 255
    bad = json.loads(first.read_text())
    bad["smoke"] = True
    first.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="expected smoke=False"):
        collect(tmp_path)


def test_affine_band_and_errors():
    x = np.arange(30.).reshape(10, 3)
    np.testing.assert_allclose(d.affine(x, x[:, :2])(x), x[:, :2], atol=1e-5)
    history = jnp.arange(12).reshape(1, 4, 3)
    band = d.band_features(history, (0, 1))
    np.testing.assert_array_equal(band[0, 2], jnp.r_[history[0, 2], history[0, 1]])
    with pytest.raises(ValueError):
        d.affine(np.empty((0, 3)), np.empty((0, 2)))
    with pytest.raises(ValueError):
        d.affine(np.full((2, 3), np.nan), np.ones((2, 2)))


def test_null_and_donor_abstention():
    h = jnp.ones((2, 3))
    direction = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    delta = d.null_write(h, direction, lambda x: x[:1])
    np.testing.assert_allclose(delta[:, 0], 0., atol=1e-5)
    np.testing.assert_allclose(delta[:, 1:], direction[:, 1:])
    target = jnp.zeros((2, 2))
    shift, support = d.donor_write(h, target, target, jnp.empty((0, 3)), jnp.empty((0, 2)), jnp.eye(3), jnp.ones(2))
    assert not support.any()
    assert not shift.any()
    shift, support = d.donor_write(h, target, target, jnp.zeros((1, 3)), jnp.ones((1, 2)), jnp.eye(3), jnp.ones(2))
    assert not support.any()
    shift, support = d.donor_write(h, target, target, jnp.zeros((1, 3)), jnp.zeros((1, 2)), jnp.eye(3), jnp.ones(2))
    assert support.all()
    np.testing.assert_allclose(shift, -h)


def test_gradient_and_causal_training():
    h = jnp.zeros((4, 3))
    policy = lambda x: jax.nn.softmax(x[:, :2], -1)
    delta = d.gradient_write(h, jnp.ones((4, 1)), lambda x: x[:, 2:], policy, steps=100)
    assert float(jnp.mean((delta[:, 2:] - 1) ** 2)) < .05
    donor = jnp.broadcast_to(jnp.array([2., -2., 0.]), h.shape)
    u, loss = d.das(h, donor, policy, policy(donor), beta=0., rank=1, steps=100)
    np.testing.assert_allclose(u.T @ u, jnp.eye(1), atol=1e-5)
    assert float(loss) < .02
    with pytest.raises(ValueError):
        d.das(h, donor, policy, policy(donor), rank=4)


def test_mlp_and_policy_inverse():
    b = jnp.array([[[.1, .9]], [[.9, .1]], [[.4, .6]], [[.6, .4]]])
    z = 4 * b[:, 0]
    inverse = d.policy_implied(b, z, steps=200)
    np.testing.assert_allclose(inverse(z), b, atol=.06)
    probe = d.mlp(z, b[:, 0], width=8, steps=200)
    np.testing.assert_allclose(probe(z), b[:, 0], atol=.06)
    np.testing.assert_allclose(inverse.policy_map(b), jax.nn.softmax(z / 1.5), atol=.02)


def test_contextual_donor_matches_complement_and_abstains():
    h = jnp.array([[0., 3.]])
    bank = jnp.array([[2., -5.], [4., 3.], [9., 3.]])
    beliefs = jnp.array([[1., 0.], [1., 0.], [0., 1.]])
    target = jnp.array([[1., 0.]])
    basis = jnp.array([[1.], [0.]])
    args = (h, target, target, bank, beliefs, basis, jnp.zeros(2), jnp.ones(2))
    delta, support = d.contextual_donor(*args)
    assert support.all()
    np.testing.assert_allclose(delta, [[4., 0.]])
    delta, support = d.contextual_donor(*args, full=True)
    np.testing.assert_allclose(h + delta, bank[1:2])
    delta, support = d.contextual_donor(h, jnp.ones((1, 2)) * .5, target, bank, beliefs,
                                      basis, jnp.zeros(2), jnp.ones(2))
    assert not support.any()
    assert not delta.any()
    delta, support = d.contextual_donor(h, target, target, bank[:0], beliefs[:0],
                                      basis, jnp.zeros(2), jnp.ones(2))
    assert not support.any()
    assert not delta.any()
    with pytest.raises(ValueError):
        d.contextual_donor(*args, tolerance=-1)


def test_all_method_cli_and_resume(monkeypatch, tmp_path, capsys):
    from experiments.reward_factors_causal_decoding_2026_11 import run
    jobs = []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, worker, work):
            jobs.extend(work)
            return []
    monkeypatch.setattr(run, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr("sys.argv", ["run", "--test", "moving", "--method", "all", "--stage2"])
    run.main()
    assert [job[2] for job in jobs] == list(run.METHODS)
    assert all(job[3] and job[-2] for job in jobs)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run.os, "sched_setaffinity", lambda *args: None)
    (tmp_path / "results").mkdir()
    output = tmp_path / "results" / "stage2_moving_seed42_band.json"
    config = dict(test="moving", seed=42, method="band", pilot=True, beta=1., write="auto", stage2=True)
    output.write_text(json.dumps(config))
    job = next(job for job in jobs if job[2] == "band")
    run.worker(job)
    assert "SKIP" in capsys.readouterr().out
    config["pilot"] = False
    output.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="incompatible result"):
        run.worker(job)


def test_action_basis_and_behavioural_write():
    h = jnp.zeros((8, 3))
    logits = lambda x: jnp.array([2 * x[0], x[1]])
    basis, singular = d.action_basis(h, logits, 2)
    np.testing.assert_allclose(basis.T @ basis, jnp.eye(2), atol=1e-6)
    np.testing.assert_allclose(basis[2], 0., atol=1e-6)
    projected, _ = d.action_basis(h, logits, 2, exclude=np.array([[1.], [0.], [0.]]))
    assert projected.shape == (3, 1)
    np.testing.assert_allclose(projected[0], 0., atol=1e-6)
    policy = lambda a: jax.nn.softmax(a[:, :2], -1)
    target = jnp.broadcast_to(jnp.array([.9, .1]), (8, 2))
    delta = d.behavioural_write(h, target, policy, basis, beta=0., steps=100)
    assert float(d.kl(target, policy(h + delta)).mean()) < .01
    np.testing.assert_allclose(delta[:, 2], 0.)
    with pytest.raises(ValueError):
        d.action_basis(h[:0], logits, 1)
    with pytest.raises(ValueError):
        d.action_basis(h, lambda a: jnp.zeros(2), 1)


def test_seed_campaign_smokes_before_full(monkeypatch):
    from experiments.reward_factors_causal_decoding_2026_11 import seed_campaign
    commands = []
    monkeypatch.setattr(seed_campaign, "prepare_archives", lambda: None)
    monkeypatch.setattr("sys.argv", ["seed_campaign", "--seed", "43"])
    monkeypatch.setattr(seed_campaign.subprocess, "run", lambda command, **kwargs: commands.append(command))
    seed_campaign.main()
    assert len(commands) == 2
    assert commands[0] == [*commands[1], "--smoke"]
    assert commands[1][commands[1].index("--seed") + 1] == "43"
    assert "--campaign" in commands[1]
