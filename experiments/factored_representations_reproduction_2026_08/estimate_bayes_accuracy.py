"""Estimate Bayes-optimal accuracy from exact environment diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_2026_08.process import (
    FACTOR_COUNTS,
    environment_config,
    joint_token_count,
)


def estimate_bayes_accuracy(
    factor_count: int,
    *,
    episodes: int,
    seed: int,
) -> dict[str, int | float]:
    """Evaluate ``argmax p(next joint token | history)`` over process histories."""

    # region agent log
    with open("/opt/cursor/logs/debug.log", "a") as log:
        log.write(json.dumps({"hypothesisId": "A,C", "location": "estimate_bayes_accuracy.py:27", "message": "estimator entry", "data": {"factor_count": factor_count, "episodes": episodes, "seed": seed}, "timestamp": __import__("time").time_ns() // 1_000_000}) + "\n")
    # endregion
    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    config = environment_config(factor_count)
    config["diagnostics"] = {"belief": True, "tokens": True, "transitions": True}
    environment = HMMEnv(config)
    # region agent log
    with open("/opt/cursor/logs/debug.log", "a") as log:
        log.write(json.dumps({"hypothesisId": "A,D", "location": "estimate_bayes_accuracy.py:40", "message": "model probability shapes", "data": {"states": environment.model.n_states, "tokens": environment.model.n_tokens, "emission_shape": list(environment.model.emission_matrix.shape)}, "timestamp": __import__("time").time_ns() // 1_000_000}) + "\n")
    # endregion
    rng = np.random.default_rng(seed)
    optimal_probabilities: list[float] = []
    first_alignment_logged = False
    try:
        for episode in range(episodes):
            _, _ = environment.reset(seed=int(rng.integers(2**32)))
            _, _, terminated, truncated, info = environment.step(0)
            if episode == 0:
                token_probabilities = (
                    info["belief_current"] @ environment.model.emission_matrix
                )
                # region agent log
                with open("/opt/cursor/logs/debug.log", "a") as log:
                    log.write(json.dumps({"hypothesisId": "A,B,D", "location": "estimate_bayes_accuracy.py:53", "message": "first retained decision distributions", "data": {"decision_step": info["decision_step"], "belief_max": float(np.max(info["belief_current"])), "token_max": float(np.max(token_probabilities)), "belief_sum": float(np.sum(info["belief_current"])), "token_sum": float(np.sum(token_probabilities)), "raw_token_current": info["raw_token_current"]}, "timestamp": __import__("time").time_ns() // 1_000_000}) + "\n")
                # endregion
            while not (terminated or truncated):
                token_probabilities = (
                    info["belief_current"] @ environment.model.emission_matrix
                )
                optimal_probabilities.append(float(np.max(token_probabilities)))
                previous_info = info
                _, _, terminated, truncated, info = environment.step(0)
                if not first_alignment_logged:
                    # region agent log
                    with open("/opt/cursor/logs/debug.log", "a") as log:
                        log.write(json.dumps({"hypothesisId": "B", "location": "estimate_bayes_accuracy.py:65", "message": "decision-to-scored-token alignment", "data": {"prediction_decision_step": previous_info["decision_step"], "predicted_raw_token_current": previous_info["raw_token_current"], "transition_step": info["transition_step"], "scored_raw_token_before": info["raw_token_before"], "aligned": previous_info["raw_token_current"] == info["raw_token_before"]}, "timestamp": __import__("time").time_ns() // 1_000_000}) + "\n")
                    # endregion
                    first_alignment_logged = True
    finally:
        environment.close()

    if not optimal_probabilities:
        raise RuntimeError("randomized episodes retained no post-BOS decisions")
    # region agent log
    with open("/opt/cursor/logs/debug.log", "a") as log:
        log.write(json.dumps({"hypothesisId": "A,C", "location": "estimate_bayes_accuracy.py:77", "message": "estimator exit", "data": {"retained_decisions": len(optimal_probabilities), "mean_hidden_state_max": float(np.mean(optimal_probabilities)), "decisions_per_episode": len(optimal_probabilities) / episodes}, "timestamp": __import__("time").time_ns() // 1_000_000}) + "\n")
    # endregion
    return {
        "factor_count": factor_count,
        "joint_token_count": joint_token_count(factor_count),
        "chance_accuracy": 1.0 / joint_token_count(factor_count),
        "estimated_bayes_accuracy": float(np.mean(optimal_probabilities)),
        "episodes": episodes,
        "retained_decisions": len(optimal_probabilities),
        "seed": seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    reports = [
        estimate_bayes_accuracy(
            factor_count,
            episodes=args.episodes,
            seed=args.seed,
        )
        for factor_count in FACTOR_COUNTS
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
