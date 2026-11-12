import json

import numpy as np

from experiments.rocksample_jax_2026_10 import belief_steering_cf as cf
from experiments.rocksample_jax_2026_10 import belief_steering_cf_confirm as confirm
from experiments.rocksample_jax_2026_10 import belief_steering_cf_report as report


class Env:
    k = 7


def test_twin_flips_only_gated_sensor_symbols():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(5, 31))
    gate = np.array([True, False, True, False, False])
    twin = cf.flip_symbols(obs, gate, Env)
    s = cf.symbol_index(Env)
    assert s == 16 and cf.check_index(Env, 6) == 30
    np.testing.assert_array_equal(twin[~gate], obs[~gate])
    np.testing.assert_array_equal(twin[gate][:, [s, s + 1]], obs[gate][:, [s + 1, s]])
    other = np.delete(np.arange(31), [s, s + 1])
    np.testing.assert_array_equal(twin[:, other], obs[:, other])


def test_steering_adds_the_projected_affine_twin_displacement():
    rng = np.random.default_rng(1)
    d, k = 6, 7
    basis = np.stack([[np.linalg.qr(rng.normal(size=(d, d)))[0].T for _ in range(k)]
                      for _ in range(cf.SITES)])
    fitted = dict(decoder=dict(prob=rng.normal(size=(cf.SITES, d, k))),
                  offset=dict(prob=rng.normal(size=(cf.SITES, k))),
                  proto=rng.normal(size=(2, cf.SITES, k, d)), basis_all=basis,
                  reflect=rng.normal(size=(2, cf.SITES, k, 2, d + 1, d)))
    rock, site, rank, strength = 3, 0, 2, 1.25
    iv = cf.intervention(fitted, rock, True, reflect="pca_all", rank=rank,
                         reflect_sites=(site,), reflect_scale=strength)
    u = basis[site, rock, :rank]
    h = rng.normal(size=d)
    for kind in range(2):
        expected = strength * (np.append(h, 1.) @ fitted["reflect"][0, site, rock, kind]) @ u.T @ u
        actual = h @ np.asarray(iv["lin"][kind, site]) + np.asarray(iv["off"][kind, site])
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert not np.asarray(iv["lin"][:, 1:]).any() and not np.asarray(iv["lam"]).any()
    assert not np.asarray(iv["patch"]).any()


def test_frozen_protocol_and_held_out_seeds():
    variants = confirm.variants()
    assert variants["steer"] == dict(reflect="pca_all", rank=4, reflect_sites=(0,), reflect_scale=1.25)
    assert variants["control"] == dict(variants["steer"], control=True)
    assert {confirm.FIT_SEED, confirm.EVAL_SEED}.isdisjoint({41101, 61307})


def test_report_pools_counts(tmp_path, monkeypatch):
    cell = dict(target=0.25, episodes=8, other_good_counts=[30, 32], other_bad_counts=[1, 16],
                mean_return=1., readout_target=.5, checks=2., visit=.5, exit=1.,
                picked_good_by_rock=[[2] * 7, [8] * 7], picked_bad_by_rock=[[1] * 7, [4] * 7])
    cells = {f"{rock}|{d}|{name}": cell for rock in range(7) for d in report.DIRECTIONS
             for name in ("intact", "steer", "control", "legacy_clamp")}
    fit = dict(diagnostics=[dict(reflect_pca_all4_r2_site0=.8)] * 7)
    for update in report.CHECKPOINTS:
        for agent in range(3):
            (tmp_path / f"agent{agent}_update{update}.json").write_text(
                json.dumps(dict(cells=cells, fit=fit)))
    assert report.pooled([dict(cells=cells)] * 3, "steer")[0].startswith("0.250 [")
    np.testing.assert_allclose(report.matrix([dict(cells=cells)] * 3, "Good→Bad", "steer"), .25)
    monkeypatch.setattr("sys.argv", ["report", str(tmp_path)])
    report.main()
    assert "30M checkpoints, three agents pooled" in (tmp_path / "tables.md").read_text()
