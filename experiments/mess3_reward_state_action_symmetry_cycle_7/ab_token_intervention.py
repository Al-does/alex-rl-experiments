"""Evaluate a trained Reward-State Quotient policy with randomized A/B inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    environment_config,
)
from harness.seeding import child_seed_sequence, seed_sequence_to_int
from learners.models.transformer import TransformerModel


def randomize_ab(observation: np.ndarray, replacement: int) -> np.ndarray:
    """Replace A or B by an independent fair draw; preserve C and action."""
    if observation.shape != (6,) or replacement not in (0, 1):
        raise ValueError("expected a six-feature observation and an A/B draw")
    token = observation[:3]
    if not np.isclose(token.sum(), 1.0) or not np.isin(token, (0.0, 1.0)).all():
        raise ValueError("expected a one-hot token")
    result = observation.copy()
    if token[2] == 0:
        result[:3] = 0
        result[replacement] = 1
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def evaluate(
    module: TransformerModel,
    *,
    episodes: int,
    batch_size: int,
    seed: int,
    identity_control: bool = False,
) -> dict[str, object]:
    if episodes <= 0 or batch_size <= 0:
        raise ValueError("episodes and batch_size must be positive")
    config = environment_config(2)
    config["randomize_first_episode_length"] = False
    horizon = int(config["episode_length"])
    module.eval().to("cpu")
    records: list[dict[str, object]] = []

    for start in range(0, episodes, batch_size):
        indices = list(range(start, min(start + batch_size, episodes)))
        pairs = [(HMMEnv(config), HMMEnv(config)) for _ in indices]
        draws = [
            np.random.default_rng(child_seed_sequence(seed, (1, index)))
            for index in indices
        ]
        episode_records = [
            {
                "episode": index,
                "env_seed": seed_sequence_to_int(
                    child_seed_sequence(seed, (0, index))
                ),
                "baseline_reward": 0.0,
                "intervention_reward": 0.0,
                "baseline_actions": [0, 0, 0],
                "intervention_actions": [0, 0, 0],
                "ab_tokens": 0,
                "changed_ab_tokens": 0,
                "closed_loop_disagreements": 0,
                "shadow_disagreements": 0,
                "shadow_disagreements_ab": 0,
                "shadow_disagreements_c": 0,
                "shadow_total_variation": 0.0,
            }
            for index in indices
        ]
        try:
            observations = []
            for pair, record in zip(pairs, episode_records):
                baseline, _ = pair[0].reset(seed=int(record["env_seed"]))
                intervened, _ = pair[1].reset(seed=int(record["env_seed"]))
                if not np.array_equal(baseline, intervened):
                    raise RuntimeError("paired environments differ at reset")
                observations.append((baseline, intervened))
            count = len(pairs)
            state = _initial_state(module, 3 * count, torch.device("cpu"))

            for _ in range(horizon):
                coins = [int(rng.integers(2)) for rng in draws]
                baseline_batch = np.stack([row[0] for row in observations])
                intervention_batch = np.stack(
                    [
                        row[1]
                        if identity_control
                        else randomize_ab(row[1], coin)
                        for row, coin in zip(observations, coins)
                    ]
                )
                shadow_batch = np.stack(
                    [
                        row[0]
                        if identity_control
                        else randomize_ab(row[0], coin)
                        for row, coin in zip(observations, coins)
                    ]
                )
                inputs = torch.from_numpy(
                    np.concatenate(
                        (baseline_batch, intervention_batch, shadow_batch)
                    )
                )
                embeddings, state = module.encode_step(inputs, state)
                logits = module.action_distribution_inputs(embeddings)
                actions = logits.argmax(dim=-1).cpu().numpy()
                probabilities = logits.softmax(dim=-1).cpu().numpy()
                variation = 0.5 * np.abs(
                    probabilities[:count] - probabilities[2 * count :]
                ).sum(axis=1)

                next_observations = []
                for i, (pair, record) in enumerate(zip(pairs, episode_records)):
                    original_token = int(baseline_batch[i, :3].argmax())
                    if original_token != 2:
                        record["ab_tokens"] += 1
                        record["changed_ab_tokens"] += int(
                            not identity_control
                            and shadow_batch[i, :3].argmax() != original_token
                        )
                    mismatch = int(actions[i] != actions[2 * count + i])
                    record["shadow_disagreements"] += mismatch
                    record[
                        "shadow_disagreements_c"
                        if original_token == 2
                        else "shadow_disagreements_ab"
                    ] += mismatch
                    record["shadow_total_variation"] += float(variation[i])
                    record["closed_loop_disagreements"] += int(
                        actions[i] != actions[count + i]
                    )
                    record["baseline_actions"][int(actions[i])] += 1
                    record["intervention_actions"][int(actions[count + i])] += 1
                    original, reward, _, truncated, _ = pair[0].step(
                        int(actions[i])
                    )
                    changed, changed_reward, _, changed_truncated, _ = pair[1].step(
                        int(actions[count + i])
                    )
                    record["baseline_reward"] += float(reward)
                    record["intervention_reward"] += float(changed_reward)
                    if truncated != changed_truncated:
                        raise RuntimeError("paired episode lengths differ")
                    next_observations.append((original, changed))
                observations = next_observations
            records.extend(episode_records)
        finally:
            for pair in pairs:
                pair[0].close()
                pair[1].close()

    baseline = np.array([row["baseline_reward"] for row in records]) / horizon
    changed = np.array([row["intervention_reward"] for row in records]) / horizon
    differences = changed - baseline
    bootstrap = np.random.default_rng(
        child_seed_sequence(seed, (2,))
    ).integers(episodes, size=(10_000, episodes))
    interval = np.quantile(differences[bootstrap].mean(axis=1), [0.025, 0.975])
    total_actions = np.stack(
        [row["baseline_actions"] for row in records]
    ).sum(axis=0)
    changed_actions = np.stack(
        [row["intervention_actions"] for row in records]
    ).sum(axis=0)
    ab_count = sum(row["ab_tokens"] for row in records)
    return {
        "protocol": {
            "condition": "Reward-State Quotient",
            "policy": "greedy",
            "intervention": (
                "identity control" if identity_control
                else "independent fair A/B draw at each decision; C and previous action preserved"
            ),
            "shadow": "randomized observations on baseline trajectory; previous executed baseline actions preserved",
            "environment": "paired independent copies initialized with the same seed per episode",
            "episodes": episodes,
            "horizon": horizon,
            "evaluation_seed": seed,
            "bootstrap": "10000 episode-paired percentile resamples, fixed policy",
        },
        "summary": {
            "baseline_reward_fraction": float(baseline.mean()),
            "intervention_reward_fraction": float(changed.mean()),
            "paired_reward_difference": float(differences.mean()),
            "paired_reward_difference_ci95": interval.tolist(),
            "baseline_action_fractions": (total_actions / total_actions.sum()).tolist(),
            "intervention_action_fractions": (
                changed_actions / changed_actions.sum()
            ).tolist(),
            "ab_resample_change_fraction": (
                sum(row["changed_ab_tokens"] for row in records) / ab_count
            ),
            "closed_loop_greedy_disagreement_fraction": (
                sum(row["closed_loop_disagreements"] for row in records)
                / (episodes * horizon)
            ),
            "shadow_greedy_disagreement_fraction": (
                sum(row["shadow_disagreements"] for row in records)
                / (episodes * horizon)
            ),
            "shadow_greedy_disagreement_fraction_on_ab": (
                sum(row["shadow_disagreements_ab"] for row in records) / ab_count
            ),
            "shadow_greedy_disagreement_fraction_on_c": (
                sum(row["shadow_disagreements_c"] for row in records)
                / (episodes * horizon - ab_count)
            ),
            "shadow_mean_total_variation": (
                sum(row["shadow_total_variation"] for row in records)
                / (episodes * horizon)
            ),
        },
        "episodes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--identity-control", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(args.threads)
    module = load_module_only(args.checkpoint)
    if not isinstance(module, TransformerModel):
        raise TypeError("checkpoint must contain a TransformerModel")
    manifest = json.loads(args.run_manifest.read_text())
    if manifest["runtime"]["seed"] != 42 or "variant_2_ctx32_ent" not in manifest["run_id"]:
        raise ValueError("manifest must describe Reward-State Quotient seed 42")
    report = evaluate(
        module,
        episodes=args.episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        identity_control=args.identity_control,
    )
    report["provenance"] = {
        "training_run_id": manifest["run_id"],
        "training_source_commit": manifest["git"]["experiment_repository"]["commit"],
        "training_library_commit": manifest["git"]["library"]["commit"],
        "run_manifest_sha256": _sha256(args.run_manifest),
        "module_state_sha256": _sha256(args.checkpoint / "module_state.pkl"),
        "checkpoint_name": Path(manifest["trials"][0]["checkpoint"]).name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
