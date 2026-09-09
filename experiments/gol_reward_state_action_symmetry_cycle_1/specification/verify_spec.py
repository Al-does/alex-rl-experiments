"""Independent implementation/structural checks for the frozen HMM.

Run: python verify_spec.py. Writes explicit JSON kernels and verification.json.
No numerical POMDP optimality or neural-training claim is made.
"""
import itertools
import json
from pathlib import Path
import unittest
import numpy as np
from four_action_hmm import (ACTIONS, AGGREGATION, EDGE_TOKENS, SPEEDS,
    ControlledHMM, build_model, stationary_distribution)


def full_state_oracle(model):
    P, G = model.P, model.G
    n = len(model.states); E = model.states.index("E")
    best = (-1., None, None)
    for policy in itertools.product(range(4), repeat=n):
        pi = stationary_distribution(P[np.array(policy), np.arange(n)])
        if pi[E] > best[0]:
            best = (float(pi[E]), list(policy), pi)
    rho, policy, pi = best
    K = P[np.array(policy), np.arange(n)]
    system = np.block([[np.eye(n) - K, np.ones((n, 1))],
                       [np.eye(n)[[-1]], np.zeros((1, 1))]])
    bias = np.linalg.solve(system, np.r_[G[np.arange(n), policy], 0.])[:-1]
    advantages = G + (P @ bias).T - rho - bias[:, None]
    assert np.max(advantages) < 1e-10
    return dict(occupancy=rho, policy_ids=policy,
                policy_names=[ACTIONS[a] for a in policy],
                stationary=pi.tolist(), average_reward_bias=bias.tolist(),
                full_information_one_step_deviation_advantages=advantages.tolist())


def structural_checks(model):
    n = len(model.states); T, G, O = model.T, model.G, model.O
    O_all = O.transpose(1, 0, 2).reshape(n, -1)
    all_features = np.c_[np.ones(n), G, O_all]
    delta = np.zeros(n); delta[-2:] = [1., -1.]
    assert np.max(abs(delta @ all_features)) < 1e-12
    result = dict(
        min_self_loop=float(np.diagonal(model.P, axis1=1, axis2=2).min()),
        G_rank=int(np.linalg.matrix_rank(G)),
        normalized_reward_token_rank=int(np.linalg.matrix_rank(all_features)),
        common_null_direction=delta.tolist(),
        common_null_error=float(abs(delta @ all_features).max()),
        full_information_oracle=full_state_oracle(model),
        constant_action_occupancies=[float(stationary_distribution(P)[n-2]) for P in model.P])
    if model.variant == 2 and n == 4:
        coarse = model.coarse()
        result['exact_quotient_error'] = float(abs(T @ AGGREGATION - AGGREGATION @ coarse.T).max())
    if n == 3:
        # A fixed-action witness of infinite observation Markov order.
        U = T[2, 1]; one = np.ones(3)
        observable = np.c_[one, U @ one, U @ U @ one]
        initial = stationary_distribution(model.P[2]); prefixes = []
        for word in [(0,), (1,), (0, 0)]:
            b = initial.copy()
            for x in word:
                b = model.update(b, 2, x)
            prefixes.append(b)
        assert abs(np.linalg.det(U)) > 1e-12
        assert np.linalg.matrix_rank(observable) == np.linalg.matrix_rank(prefixes) == 3
        result['infinite_order_witness'] = dict(fixed_action="aE",
            token1_matrix=U.tolist(), det_token1=float(np.linalg.det(U)),
            observable_rank=3, prefix_belief_rank=3,
            prefixes=[[0], [1], [0, 0]], prefix_beliefs=np.asarray(prefixes).tolist())
    return result


class ReferenceChecks(unittest.TestCase):
    def test_probabilities_labels_and_prior(self):
        for variant, speed in itertools.product((2, 3), SPEEDS):
            m = build_model(variant, speed)
            np.testing.assert_allclose(m.T.sum((1, 3)), 1.)
            np.testing.assert_allclose(m.O.sum(2), 1.)
            np.testing.assert_allclose(m.reset_prior @ m.P.mean(0), m.reset_prior)
            self.assertGreaterEqual(m.T.min(), 0.)
            self.assertGreaterEqual(np.diagonal(m.P, axis1=1, axis2=2).min(), .15)
            for a, i, j in itertools.product(range(4), repeat=3):
                if m.P[a, i, j] > 0:
                    self.assertEqual(np.count_nonzero(m.T[a, :, i, j]), 1)
                    self.assertEqual(np.argmax(m.T[a, :, i, j]), EDGE_TOKENS[i, j])

    def test_filter_against_hidden_path_enumeration(self):
        # Independent definition: explicitly sum all hidden paths for an
        # action/token history, using P and deterministic edge labels.
        actions, tokens = (0, 3, 2, 1), (0, 1, 1, 0)
        for variant, speed in itertools.product((2, 3), SPEEDS):
            m = build_model(variant, speed); truth = np.zeros(4)
            for path in itertools.product(range(4), repeat=5):
                weight = m.reset_prior[path[0]]
                for t, (a, x) in enumerate(zip(actions, tokens)):
                    i, j = path[t:t+2]
                    weight *= m.P[a, i, j] * (EDGE_TOKENS[i, j] == x)
                truth[path[-1]] += weight
            b = m.reset_prior.copy()
            for a, x in zip(actions, tokens):
                b = m.update(b, a, x)
            np.testing.assert_allclose(b, truth / truth.sum(), atol=1e-13)

    def test_quotient_commutes_with_filter(self):
        for speed in SPEEDS:
            fine = build_model(2, speed); coarse = fine.coarse()
            np.testing.assert_allclose(fine.T @ AGGREGATION, AGGREGATION @ coarse.T, atol=1e-14)
            b = fine.reset_prior.copy(); c = coarse.reset_prior.copy()
            rng = np.random.default_rng(120)
            for _ in range(200):
                a, x = int(rng.integers(4)), int(rng.integers(2))
                b, c = fine.update(b, a, x), coarse.update(c, a, x)
                np.testing.assert_allclose(b @ AGGREGATION, c, atol=1e-13)
                np.testing.assert_allclose(fine.predict_rewards(b), coarse.predict_rewards(c), atol=1e-13)
            with self.assertRaises(ValueError):
                build_model(3, speed).coarse()

    def test_reward_timing_and_joint_sampling(self):
        m = build_model(3, "half"); env = ControlledHMM(m, seed=21)
        env.reset(prior=np.array([0., 0., 1., 0.]))
        self.assertEqual(env.hidden_state_for_evaluation, 2)
        for _ in range(1000):
            source = env.hidden_state_for_evaluation
            token, reward = env.step(3)
            dest = env.hidden_state_for_evaluation
            self.assertEqual(token, EDGE_TOKENS[source, dest])
            self.assertEqual(reward, float(dest == 2))
            if source == 2:
                self.assertEqual(reward, float(token == 0))
            if source == 3 and token == 0:
                self.assertEqual(reward, 0.)
        # An S->M transition in the quotient can carry either token.
        c = build_model(2, "half", coarse=True)
        self.assertAlmostEqual(c.T[3, 0, 2, 0], .03)
        self.assertAlmostEqual(c.T[3, 1, 2, 0], .07)

    def test_analytic_posteriors_and_invalid_inputs(self):
        m = build_model(3, "half"); s = np.array([0., 0., 0., 1.])
        np.testing.assert_allclose(m.update(s, 3, 0), np.array([.03, 0, 0, .435]) / .465)
        np.testing.assert_allclose(m.update(s, 3, 1), np.array([0, .07, .465, 0]) / .535)
        for a, x in [(4, 0), (-1, 0), (0, 2), (True, 0)]:
            with self.assertRaises(ValueError):
                m.update(s, a, x)


def export():
    models, verification = {}, {}
    for speed in SPEEDS:
        for variant in (2, 3):
            key = f"{speed}_variant{variant}"
            model = build_model(variant, speed)
            models[key] = model.as_dict(); verification[key] = structural_checks(model)
            if variant == 2:
                key += "_coarse"
                models[key] = model.coarse().as_dict()
                verification[key] = structural_checks(model.coarse())
    payload = dict(spec_version="1.0", status="frozen environment specification",
        timing="b_t -> a_t -> (s_{t+1}, x_{t+1}, r_{t+1}=1[s_{t+1}=E])",
        actor_inputs="past actions and observed tokens; excludes rewards and hidden-state labels",
        initial_observation=None, terminal_states=[], default_planning_discount=.99,
        edge_tokens=EDGE_TOKENS.tolist(), aggregation_matrix=AGGREGATION.tolist(), models=models)
    Path("hmm_values.json").write_text(json.dumps(payload, indent=2) + "\n")
    Path("verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


if __name__ == "__main__":
    run = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceChecks))
    if not run.wasSuccessful():
        raise SystemExit(1)
    for key, result in export().items():
        print(key, "rank", result['normalized_reward_token_rank'],
              "oracle", result['full_information_oracle']['policy_names'])
