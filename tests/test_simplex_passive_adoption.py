from __future__ import annotations

import numpy as np
from analysis.probes import predictive_belief_sequence
from analysis.simplex import build_simplex_run
from envs.hmm import HMMEnv

from experiments.nonergodic_mess3_token_guess_cycle_1.analysis import _target_adapter
from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    CONTEXT_LENGTH,
    environment_config,
    nonergodic_mess3_model,
)


def test_passive_complete_histories_match_adapter_and_ignore_guesses():
    model = nonergodic_mess3_model()
    histories, beliefs = [], []
    for guess in (0, 2):
        env = HMMEnv({
            **environment_config(),
            "diagnostics": {"belief": True, "state": True, "tokens": True, "transitions": True},
        })
        try:
            observation, info = env.reset(seed=57)
            observations, infos, tokens = [observation], [info], []
            for _ in range(CONTEXT_LENGTH - 1):
                observation, _, terminated, truncated, info = env.step(guess)
                observations.append(observation)
                infos.append(info)
                tokens.append(int(info["visible_token_current"]))
            assert terminated or truncated
        finally:
            env.close()
        source = predictive_belief_sequence(
            model.initial_distribution, model.edge_transition_matrices[tokens],
        )
        targets = _target_adapter(np.stack(observations), infos, np.arange(CONTEXT_LENGTH))
        np.testing.assert_allclose(source, targets["weighted_belief"], atol=1e-12)
        np.testing.assert_allclose(
            source @ model.transition_matrix, np.stack([info["belief_current"] for info in infos]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            source @ model.emission_matrix, targets["next_token_distribution"], atol=1e-12,
        )
        histories.append(["BOS", *[str(token) for token in tokens]])
        beliefs.append(source)
    assert histories[0] == histories[1]
    np.testing.assert_array_equal(beliefs[0], beliefs[1])
    targets = np.stack(beliefs)
    run = build_simplex_run(
        name="Passive adapter check",
        targets=targets,
        predictions={"identity target control": {"belief": targets}},
        components={"A": [0, 1, 2], "B": [3, 4, 5]},
        state_labels=model.state_labels,
        primary_site="belief",
        tokens=histories,
        cloud_rows=[0, 1, 127, 255],
        example_episodes=[0, 1],
    )
    assert len(run["sequences"][0]["targets"]) == CONTEXT_LENGTH
    assert run["sequences"][0]["notes"] == [""] * CONTEXT_LENGTH
    assert run["metrics"]["identity target control"]["belief"]["r_squared"] == 1
