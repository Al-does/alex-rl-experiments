"""Causal belief-subspace interventions on a frozen Joint Reward policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from experiments.two_factor_reward_state_PPO_cycle_2.analysis import (
    collect_probe_data,
)
from experiments.two_factor_reward_state_PPO_cycle_2.process import (
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import ACTION_PAIRS


MODES = (
    "intact", "erase_2", "swap_2", "opposite_2", "opposite_2_clamp_1",
    "erase_1", "random_direction", "random_actions",
)
PAIR_TO_ACTION = {pair: index for index, pair in enumerate(ACTION_PAIRS)}


def affine_fit(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack((features, np.ones(len(features))))
    fitted = np.linalg.lstsq(x, targets, rcond=1e-10)[0]
    return fitted[:-1], fitted[-1]


def r_squared(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = np.sum((prediction - truth) ** 2)
    variance = np.sum((truth - truth.mean(0)) ** 2)
    return float(1 - error / variance)


def intervention_delta(
    residual: np.ndarray,
    decoder: np.ndarray,
    bias: np.ndarray,
    inverse: np.ndarray,
    mean: np.ndarray,
    mode: str,
    random_direction: np.ndarray,
    embedding: np.ndarray | None = None,
) -> np.ndarray:
    decoded = residual @ decoder + bias
    target = decoded.copy()
    if mode == "erase_2" or mode == "random_direction":
        target[:, 2:] = mean[2:]
    elif mode == "swap_2":
        second = np.column_stack((decoded[:, 2:], 1 - decoded[:, 2:].sum(axis=1)))
        target[:, 2:] = second[:, [1, 0]]
    elif mode in ("opposite_2", "opposite_2_clamp_1"):
        target[:, 2] = (decoded[:, 3] > decoded[:, 2]).astype(np.float64)
        target[:, 3] = 1 - target[:, 2]
    elif mode == "erase_1":
        target[:, :2] = mean[:2]
    else:
        return np.zeros_like(residual)
    delta = (target - decoded) @ (inverse if embedding is None else embedding)
    if mode == "random_direction":
        delta = np.linalg.norm(delta, axis=1, keepdims=True) * random_direction
    return delta


@torch.inference_mode()
def evaluate(
    module: torch.nn.Module,
    *,
    decoder: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    embedding: np.ndarray | None,
    episodes: int,
    strength: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    inverse = np.linalg.pinv(decoder, rcond=1e-10)
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(decoder.shape[0])
    direction -= direction @ decoder @ inverse
    direction /= np.linalg.norm(direction)
    seeds = rng.integers(0, 2**31, size=episodes)
    action_seeds = rng.integers(0, 2**31, size=episodes)
    config = environment_config("reward_both")
    config["randomize_first_episode_length"] = False
    config["diagnostics"] = {"state": True}
    report: dict[str, dict[str, object]] = {}
    for mode in MODES:
        totals = np.zeros((episodes, 2), dtype=np.float64)
        fractions = np.zeros((2, 3), dtype=np.int64)
        changes = np.zeros((2, 2), dtype=np.float64)
        actions_changed = 0
        for offset in range(0, episodes, 16):
            count = min(16, episodes - offset)
            envs = [HMMEnv(config) for _ in range(count)]
            try:
                observations, infos = zip(
                    *(env.reset(seed=int(seeds[offset + index]))
                      for index, env in enumerate(envs))
                )
                observations = np.stack(observations)
                infos = list(infos)
                generators = [
                    np.random.default_rng(int(value))
                    for value in action_seeds[offset:offset + count]
                ]
                state = _initial_state(module, count, torch.device("cpu"))
                for _ in range(1024):
                    residual, state = module.encode_step_pre_final_norm(
                        torch.as_tensor(observations, dtype=torch.float32), state
                    )
                    base = residual.cpu().numpy()
                    shift = strength * intervention_delta(
                        base, decoder, bias, inverse, mean, mode, direction, embedding
                    )
                    logits = module.action_distribution_inputs(
                        module.encoder.final_norm(
                            residual + torch.as_tensor(shift, dtype=residual.dtype)
                        )
                    )
                    actions = logits.argmax(dim=-1).cpu().numpy()
                    if mode == "random_actions":
                        actions = np.asarray([
                            generator.integers(9) for generator in generators
                        ])
                    else:
                        unmodified = module.action_distribution_inputs(
                            module.encoder.final_norm(residual)
                        ).argmax(dim=-1).cpu().numpy()
                        if mode == "opposite_2_clamp_1":
                            actions = np.asarray([
                                PAIR_TO_ACTION[
                                    (ACTION_PAIRS[int(original)][0],
                                     ACTION_PAIRS[int(edited)][1])
                                ]
                                for original, edited in zip(unmodified, actions)
                            ])
                        actions_changed += int(np.count_nonzero(actions != unmodified))
                    changes += np.abs(shift @ decoder).sum(axis=0).reshape(2, 2)
                    next_obs = []
                    for index, env in enumerate(envs):
                        latent = int(infos[index]["state_current"])
                        totals[offset + index] += (latent // 3 == 2, latent % 3 == 2)
                        pair = ACTION_PAIRS[int(actions[index])]
                        fractions[0, pair[0]] += 1
                        fractions[1, pair[1]] += 1
                        observation, _, terminated, truncated, info = env.step(
                            int(actions[index])
                        )
                        if terminated or truncated:
                            if int(info["decision_step"]) != 1024:
                                raise ValueError(
                                    "evaluation requires full 1024-step episodes"
                                )
                        next_obs.append(observation)
                        infos[index] = info
                    observations = np.stack(next_obs)
            finally:
                for env in envs:
                    env.close()
        episode_occupancies = totals / 1024
        report[mode] = {
            "factor_occupancy": episode_occupancies.mean(axis=0).tolist(),
            "factor_occupancy_se": (
                (episode_occupancies.std(axis=0, ddof=1) / np.sqrt(episodes)).tolist()
                if episodes > 1 else None
            ),
            "factor_action_fractions": (fractions / (episodes * 1024)).tolist(),
            "fraction_actions_changed": (
                actions_changed / (episodes * 1024)
                if mode != "random_actions" else None
            ),
            "mean_absolute_decoded_shift": (changes / (episodes * 1024)).tolist(),
        }
    return report


def run(checkpoint: Path, output: Path, *, steps: int, episodes: int, seed: int,
        strength: float, mapping: str = "decoder_inverse",
        evaluation_seed: int | None = None) -> None:
    if mapping not in ("decoder_inverse", "belief_embedding"):
        raise ValueError("unknown belief-to-activation mapping")
    if evaluation_seed is None:
        evaluation_seed = seed + 2
    torch.set_num_threads(1)
    module = load_module_only(checkpoint).to("cpu").eval()
    train = collect_probe_data(module, condition="reward_both", n_steps=steps,
                               seed=np.random.SeedSequence([seed, 0]), device=torch.device("cpu"))
    test = collect_probe_data(module, condition="reward_both", n_steps=steps,
                              seed=np.random.SeedSequence([seed, 1]), device=torch.device("cpu"))
    x_train = np.asarray(train.activations, dtype=np.float64)
    x_test = np.asarray(test.activations, dtype=np.float64)
    y_train = np.asarray(train.factor_beliefs[:, :, :2].reshape(-1, 4), dtype=np.float64)
    y_test = np.asarray(test.factor_beliefs[:, :, :2].reshape(-1, 4), dtype=np.float64)
    decoder, bias = affine_fit(x_train, y_train)
    embedding, embedding_bias = affine_fit(y_train, x_train)
    first_subspace, _ = np.linalg.qr(decoder[:, :2])
    second_subspace, _ = np.linalg.qr(decoder[:, 2:])
    embedding_first, _ = np.linalg.qr(embedding[:2].T)
    embedding_second, _ = np.linalg.qr(embedding[2:].T)
    estimated = x_test @ decoder + bias
    predicted_residual = y_test @ embedding + embedding_bias
    report = {
        "checkpoint": str(checkpoint),
        "module_state_sha256": hashlib.sha256((checkpoint / "module_state.pkl").read_bytes()).hexdigest(),
        "seed": seed,
        "fit_steps": steps,
        "test_steps": steps,
        "fit_test_independent_rollouts": True,
        "probe_warmup_steps": 8,
        "representation": "final_block_residual_before_final_layer_norm",
        "method": "Affine activations-to-factor-beliefs decoder on both factors. Belief embedding fits beliefs-to-activations and adds its factor contribution; decoder inverse uses the minimum-norm right-inverse of the four-column decoder to adjust one decoded factor while preserving the other. Current decoded beliefs, not environment diagnostics, set the intervention. The factor-2 swap exchanges decoded nonreward states 0 and 1; opposite_2 targets the opposite nonreward state with probability one. Random direction is orthogonal to decoder columns with matched per-step shift norm. The clamp control combines the intact policy's factor-1 action with the steered policy's factor-2 action; it is a downstream action intervention, unlike the other conditions.",
        "mapping": mapping,
        "probe_r2": [r_squared(estimated[:, i:i+2], y_test[:, i:i+2]) for i in (0, 2)],
        "belief_to_activation_r2": r_squared(predicted_residual, x_test),
        "decoder_condition_number": float(np.linalg.cond(decoder)),
        "factor_subspace_principal_cosines": np.linalg.svd(
            first_subspace.T @ second_subspace, compute_uv=False
        ).tolist(),
        "embedding_factor_subspace_principal_cosines": np.linalg.svd(
            embedding_first.T @ embedding_second, compute_uv=False
        ).tolist(),
        "embedding_top_channels_by_factor": [
            np.argsort(-np.linalg.norm(embedding[i:i+2], axis=0))[:8].tolist()
            for i in (0, 2)
        ],
        "decoder_weights": decoder.tolist(),
        "decoder_bias": bias.tolist(),
        "embedding_weights": embedding.tolist(),
        "embedding_bias": embedding_bias.tolist(),
        "fit_belief_mean": y_train.mean(axis=0).tolist(),
        "strength": strength,
        "evaluation_episodes": episodes,
        "evaluation_horizon": 1024,
        "evaluation_seed": evaluation_seed,
        "behavior": evaluate(module, decoder=decoder, bias=bias, mean=y_train.mean(axis=0),
                             embedding=embedding if mapping == "belief_embedding" else None,
                             episodes=episodes, strength=strength, seed=evaluation_seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"probe_r2": report["probe_r2"], "behavior": report["behavior"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mapping", choices=("decoder_inverse", "belief_embedding"),
                        default="decoder_inverse")
    args = parser.parse_args()
    run(args.checkpoint, args.output, steps=args.steps, episodes=args.episodes,
        seed=args.seed, strength=args.strength, mapping=args.mapping,
        evaluation_seed=args.evaluation_seed)


if __name__ == "__main__":
    main()
