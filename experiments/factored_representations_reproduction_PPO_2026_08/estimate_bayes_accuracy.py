"""Estimate Bayes-optimal accuracy from exact environment diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.process import (
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

    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    config = environment_config(factor_count)
    config["diagnostics"] = {"belief": True}
    environment = HMMEnv(config)
    rng = np.random.default_rng(seed)
    optimal_probabilities: list[float] = []
    try:
        for _ in range(episodes):
            _, _ = environment.reset(seed=int(rng.integers(2**32)))
            _, _, terminated, truncated, info = environment.step(0)
            while not (terminated or truncated):
                token_probabilities = (
                    info["belief_current"] @ environment.model.emission_matrix
                )
                optimal_probabilities.append(float(np.max(token_probabilities)))
                _, _, terminated, truncated, info = environment.step(0)
    finally:
        environment.close()

    if not optimal_probabilities:
        raise RuntimeError("randomized episodes retained no post-BOS decisions")
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
