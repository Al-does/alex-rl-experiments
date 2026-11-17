import unittest
import json
import jax
import numpy as np

from .analyze import actor_arrays, discover, within
from .paper_constraints import constrain_predictions
from .paper_coarse import merged_filter
from .paper_probe import coarse_filter
from .paper_within import affine, refit
from .report import metrics


class AdapterTests(unittest.TestCase):
    def test_actor_exact_keys_shapes_and_value_only_removal(self):
        template = {"policy": {"w": jax.numpy.ones((64, 3)), "b": jax.numpy.zeros(3)}}
        saved = {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_leaves_with_path(template)}
        actor_arrays(saved, template, "sac_autoalpha")
        ppo = {**saved, "['value']['w']": np.zeros((64, 1), np.float32), "['value']['b']": np.zeros(1, np.float32)}
        self.assertEqual(set(actor_arrays(ppo, template, "ppo")), {"policy"})
        for bad, algorithm in (({}, "ppo"), (ppo, "sac_autoalpha"), (saved, "ppo"),
                               ({**ppo, "other": np.zeros(1)}, "ppo"),
                               ({**ppo, "['value']['w']": np.zeros((1, 64))}, "ppo"),
                               ({**ppo, "['policy']['w']": np.full((64, 3), np.nan, np.float32)}, "ppo")):
            with self.assertRaises(ValueError):
                actor_arrays(bad, template, algorithm)

    def test_inventory_exactly_30(self):
        self.assertEqual(len(discover()), 30)

    def test_constraints_and_invalid_empty(self):
        np.testing.assert_allclose(constrain_predictions(np.array([[.8, .8, -.2]]), "belief"), [[.5, .5, 0]])
        np.testing.assert_array_equal(constrain_predictions(np.array([[-3.], [2.]]), "delta"), [[-1], [1]])
        for a in (np.empty((0, 3)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                constrain_predictions(a, "belief")

    def test_filter_equivalence_and_invalid_empty(self):
        from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
        process = dict(initial=[1/3]*3, emission=environment.emission_matrix().tolist(),
                       transitions=environment.action_transitions(2).tolist())
        obs = np.zeros((2, 8, 6)); obs[..., 2] = 1; obs[:, 1:, 3] = 1
        np.testing.assert_allclose(coarse_filter(obs, process), merged_filter(obs, process)[..., 2], atol=1e-12)
        with self.assertRaises(ValueError):
            merged_filter(np.empty((0, 8, 6)), process)

    def test_affine_and_missing_class(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(50, 4)); y = x[:, :1]*2+1
        np.testing.assert_allclose(affine(x, y, x), y, atol=1e-6)
        pred = refit(x, y, x, np.zeros(50, int), np.ones(50, int))
        self.assertTrue(np.isnan(pred).all())

    def test_within_full_headblind_and_logits(self):
        rng = np.random.default_rng(2)
        x = [rng.normal(size=(60, 5)), rng.normal(size=(80, 5))]
        b = [np.column_stack([h[:, 0], h[:, 1], h[:, 2]]) for h in x]
        classes = [np.arange(len(h)) % 3 for h in x]
        params = {"policy": {"w": rng.normal(size=(5, 3)), "b": np.zeros(3)}}
        result = within(params, x, b, classes)
        self.assertEqual(result["head_null_dimension"], 3)
        self.assertGreater(result["metrics"]["full_h"]["belief"]["r2"], .999)
        self.assertEqual(result["metrics"]["full_h"]["belief"]["excluded_rows"], 0)
        json.dumps(result, allow_nan=False)

    def test_summary_retains_trivial_coverage_but_rejects_missing_required_classes(self):
        value = {"r2": .75, "excluded_rows": 1}
        run = {"variant": 1, "run_id": "test", "scalars": {"belief": {"error": .2}},
               "trained": {"metrics": {"full_h": {"belief": value}}},
               "matched_init": {"metrics": {"full_h": {"belief": value}}}}
        self.assertEqual(metrics(run)["within_trained_full_h_belief"], .75)
        run["variant"] = 2
        with self.assertRaises(ValueError):
            metrics(run)


if __name__ == "__main__":
    unittest.main()
