from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.probes import (
    cluster_bootstrap_statistics,
    conditional_mse_metrics,
    fit_affine_probe,
    global_mse_metrics,
    mean_squared_error,
    percentile_interval,
    probe_predict,
    r2_score,
)
from analysis.rollouts import collect_batched_rollout_data
from envs.gol.model import AGGREGATION, controlled_kernels
from envs.hmm import HMMEnv
from envs.hmm.env import ObservationConfig
from experiments.gol_reward_state_action_symmetry_cycle_1.process import environment_config
from harness.seeding import named_seed_sequences, seed_sequence_to_int


SMOKE_PROBE_STEPS = 256
FULL_PROBE_STEPS = 10_000
RIDGE = 1e-6
STREAM_KEYS = {
    "fit": (710,), "test": (711,),
    "shuffle_fit": (712,), "shuffle_test": (713,),
    "permuted_labels": (714,), "bootstrap": (715,), "lookup_controller": (716,),
}


@dataclass(frozen=True)
class ProbeData:
    activations: np.ndarray
    beliefs: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_steps: np.ndarray
    env_indices: np.ndarray
    history_observations: np.ndarray
    sample_indices: np.ndarray


def belief_targets(beliefs: np.ndarray) -> dict[str, np.ndarray]:
    beliefs = np.asarray(beliefs, dtype=np.float64)
    if beliefs.ndim != 2 or beliefs.shape[1] != 4:
        raise ValueError("beliefs must have shape (N, 4) in M1,M2,E,S order")
    if not np.isfinite(beliefs).all() or not np.allclose(beliefs.sum(axis=1), 1.0):
        raise ValueError("beliefs must be finite and normalized")
    return {
        "fine_belief": beliefs,
        "coarse_belief": beliefs @ AGGREGATION,
        "m1_minus_m2": beliefs[:, 0:1] - beliefs[:, 1:2],
        "e_minus_s": beliefs[:, 2:3] - beliefs[:, 3:4],
    }


def branch_keys(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != 6:
        raise ValueError("actor observations must contain only two token and four action features")
    if not np.isin(observations, [0, 1]).all():
        raise ValueError("actor inputs must be categorical one-hot features")
    reset = (observations == 0).all(axis=1)
    if not np.all(observations[~reset, :2].sum(axis=1) == 1) or not np.all(
        observations[~reset, 2:].sum(axis=1) == 1
    ):
        raise ValueError("each non-reset observation needs one token and one previous action")
    keys = 2 * observations[:, 2:].argmax(axis=1) + observations[:, :2].argmax(axis=1)
    return np.where(reset, 8, keys)


def marginal_features(beliefs: np.ndarray, kernels: np.ndarray) -> dict[str, np.ndarray]:
    reward = np.asarray(beliefs, dtype=np.float64) @ kernels.sum(axis=1)[:, :, 2].T
    token = np.einsum("ni,axi->nax", beliefs, kernels.sum(axis=3)).reshape(len(beliefs), 8)
    return {"next_reward": reward, "next_token": token, "combined_marginals": np.column_stack((reward, token))}


def _environment(variant: int, speed: str) -> HMMEnv:
    config = environment_config(variant, speed)
    config["diagnostics"] = {"belief": True}
    env = HMMEnv(config)
    if (
        env.config.observation != ObservationConfig()
        or env.config.delay != 0 or env.config.reset_emission
        or env.config.episode_length is not None
        or env.model.n_states != 4 or env.action_space.n != 4
    ):
        env.close()
        raise ValueError("Gol probes require continuing fine-state action/token-only observations")
    return env


def _targets(observations, infos, steps):
    branch_keys(observations)
    beliefs = np.asarray([info["belief_current"] for info in infos], dtype=np.float64)
    belief_targets(beliefs)
    return {"belief": beliefs, "step": steps}


def _initial_state(module, count: int) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(np.repeat(np.asarray(value)[None], count, axis=0), device="cpu")
        for key, value in module.get_initial_state().items()
    }


def _reset_state(module, state, indices):
    fresh = _initial_state(module, len(indices))
    for key in state:
        state[key][indices] = fresh[key]
    return state


@torch.no_grad()
def collect_probe_data(
    module, *, variant: int, speed: str = "half", n_steps: int,
    seed: int | np.random.SeedSequence, warmup: int = 32, n_envs: int = 4,
) -> ProbeData:
    if n_steps <= 0 or n_envs <= 0 or warmup < 0:
        raise ValueError("positive sample/environment counts and nonnegative warmup required")
    module.to("cpu").eval()

    def step(observations, state, randomness, action_spaces):
        branch_keys(observations)
        embedding, state = module.encode_step(torch.as_tensor(observations), state)
        probabilities = module.action_distribution_inputs(embedding).softmax(dim=-1).numpy()
        actions = np.array([randomness.numpy.choice(4, p=p / p.sum()) for p in probabilities])
        return actions, state, embedding.numpy()

    total = (warmup + (n_steps + n_envs - 1) // n_envs) * n_envs
    data = collect_batched_rollout_data(
        lambda: _environment(variant, speed), step, _targets,
        n_steps=total, seed=seed, n_envs=n_envs,
        initial_state=lambda count: _initial_state(module, count),
        reset_state=lambda state, indices: _reset_state(module, state, indices),
        store_observations=True,
    )
    selected = np.arange(warmup * n_envs, warmup * n_envs + n_steps)
    return ProbeData(
        activations=data.representations[selected], beliefs=data.targets["belief"][selected],
        observations=data.observations[selected], actions=data.actions[selected].reshape(-1),
        rewards=data.rewards[selected], episode_steps=data.targets["step"][selected],
        env_indices=selected % n_envs,
        history_observations=data.observations.reshape(-1, n_envs, 6), sample_indices=selected,
    )


def shuffled_histories(histories: np.ndarray, seed) -> np.ndarray:
    histories = np.asarray(histories)
    if histories.ndim != 3 or histories.shape[2] != 6 or np.any(histories[0]):
        raise ValueError("histories must start with the empty-history reset observation")
    rng = np.random.default_rng(seed)
    shuffled = histories.copy()
    for env_index in range(histories.shape[1]):
        shuffled[1:, env_index] = histories[1:, env_index][rng.permutation(len(histories) - 1)]
    return shuffled


@torch.no_grad()
def replay_representations(module, histories: np.ndarray) -> np.ndarray:
    module.to("cpu").eval()
    state = _initial_state(module, histories.shape[1])
    representations = []
    for observations in histories:
        branch_keys(observations)
        embedding, state = module.encode_step(torch.as_tensor(observations, dtype=torch.float32), state)
        representations.append(embedding.numpy())
    return np.stack(representations).reshape(-1, embedding.shape[-1])


def branch_prediction(train_keys, test_keys, train_target):
    prediction = np.tile(train_target.mean(axis=0), (len(test_keys), 1))
    for key in np.unique(train_keys):
        prediction[test_keys == key] = train_target[train_keys == key].mean(axis=0)
    return prediction


def _affine_prediction(train_features, test_features, train_target):
    weight, bias = fit_affine_probe(train_features, train_target, ridge=RIDGE)
    return probe_predict(weight, bias, test_features)


def _metrics(predicted, target, groups):
    return {
        **global_mse_metrics(predicted, target),
        **conditional_mse_metrics(predicted, target, groups, min_group_size=1),
        "r_squared": r2_score(predicted, target),
        "sampling_distribution": "process_weighted_rollout",
    }


def _policy_metrics(beliefs, actions, rewards, kernels):
    reward_predictions = marginal_features(beliefs, kernels)["next_reward"]
    return {
        "reward_mean": float(np.mean(rewards)),
        "expected_next_reward_mean": float(reward_predictions[np.arange(len(actions)), actions].mean()),
        "action_fractions": (np.bincount(actions, minlength=4) / len(actions)).tolist(),
        "n_evaluated": len(actions),
    }


def evaluate_lookup_controller(train, *, variant, speed, n_steps, n_envs, warmup, seed):
    kernels = controlled_kernels(variant, speed)
    centroids = branch_prediction(branch_keys(train.observations), np.arange(9), train.beliefs)
    table = marginal_features(centroids, kernels)["next_reward"].argmax(axis=1)

    def step(observations, state, randomness, action_spaces):
        return table[branch_keys(observations)], None, observations

    data = collect_batched_rollout_data(
        lambda: _environment(variant, speed), step, _targets,
        n_steps=n_steps, seed=seed, n_envs=n_envs, warmup=warmup,
    )
    return {
        **_policy_metrics(data.targets["belief"], data.actions.reshape(-1), data.rewards, kernels),
        "action_table_by_branch": table.tolist(),
        "method": "Myopic lookup fitted using fit-history belief centroids and known reward kernels; unseen branches use the fit global mean. Fresh on-policy evaluation; not an optimized long-run memory-one controller.",
    }


def analyze_samples(train, test, *, shuffled_fit, shuffled_test, kernels, streams, smoke):
    train_targets, test_targets = belief_targets(train.beliefs), belief_targets(test.beliefs)
    fit_keys, test_keys = branch_keys(train.observations), branch_keys(test.observations)
    fit_marginals = marginal_features(train.beliefs, kernels)
    test_marginals = marginal_features(test.beliefs, kernels)
    features = {
        "post_final_layer_norm": (train.activations, test.activations),
        "shuffled_history_broken_alignment": (shuffled_fit, shuffled_test),
        **{name: (fit_marginals[name], test_marginals[name]) for name in fit_marginals},
        "categorical_plus_marginals": (
            np.column_stack((np.eye(9)[fit_keys], fit_marginals["combined_marginals"])),
            np.column_stack((np.eye(9)[test_keys], test_marginals["combined_marginals"])),
        ),
    }
    report = {name: {} for name in features}
    report["previous_action_latest_token"] = {}
    report["single_permuted_training_labels"] = {}
    permutation = np.random.default_rng(streams["permuted_labels"]).permutation(len(train.beliefs))
    for name, target in test_targets.items():
        for feature_name, (fit_features, test_features) in features.items():
            predicted = _affine_prediction(fit_features, test_features, train_targets[name])
            metrics = _metrics(predicted, target, test_keys)
            if feature_name == "post_final_layer_norm":
                estimates = cluster_bootstrap_statistics(
                    test.env_indices, lambda indices: mean_squared_error(predicted[indices], target[indices]),
                    n_resamples=20 if smoke else 200, seed=seed_sequence_to_int(streams["bootstrap"]),
                )
                metrics["mse_environment_bootstrap_ci95"] = list(percentile_interval(estimates))
            report[feature_name][name] = metrics
        baseline = branch_prediction(fit_keys, test_keys, train_targets[name])
        report["previous_action_latest_token"][name] = _metrics(baseline, target, test_keys)
        permuted = _affine_prediction(train.activations, test.activations, train_targets[name][permutation])
        report["single_permuted_training_labels"][name] = _metrics(permuted, target, test_keys)
    return report


def load_checkpoint_module(checkpoint: Path):
    from ray.rllib.core.rl_module.rl_module import RLModule

    nested = checkpoint / "learner_group" / "learner" / "rl_module" / "default_policy"
    module_path = nested if nested.is_dir() else checkpoint
    return RLModule.from_checkpoint(str(module_path)).to("cpu").eval()


def _json_native(value):
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def analyze_checkpoint(
    context, *, checkpoint: Path, variant: int, speed: str = "half", agent_steps: int = 0,
    checkpoint_label: str = "final", training_iteration: int = 0,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("checkpoint probing requires a resolved seed")
    if not checkpoint_label or Path(checkpoint_label).name != checkpoint_label:
        raise ValueError("checkpoint_label must be a nonempty filename component")
    kernels = controlled_kernels(variant, speed)
    streams = named_seed_sequences(context.seed, STREAM_KEYS)
    n_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_STEPS
    n_envs, warmup = (4, 32) if context.smoke else (16, 64)
    module = load_checkpoint_module(Path(checkpoint))
    common = dict(variant=variant, speed=speed, n_steps=n_steps, n_envs=n_envs, warmup=warmup)
    train = collect_probe_data(module, seed=streams["fit"], **common)
    test = collect_probe_data(module, seed=streams["test"], **common)
    shuffled = [
        replay_representations(module, shuffled_histories(data.history_observations, streams[key]))[data.sample_indices]
        for data, key in ((train, "shuffle_fit"), (test, "shuffle_test"))
    ]
    fits = analyze_samples(
        train, test, shuffled_fit=shuffled[0], shuffled_test=shuffled[1],
        kernels=kernels, streams=streams, smoke=context.smoke,
    )
    lookup = evaluate_lookup_controller(train, seed=streams["lookup_controller"], **common)
    filename = f"probe_{checkpoint_label}.json"
    report = _json_native({
        "schema_version": 1, "variant": variant, "speed": speed,
        "checkpoint_label": checkpoint_label, "agent_steps": agent_steps,
        "training_iteration": training_iteration, "n_fit": len(train.beliefs), "n_test": len(test.beliefs),
        "probe_fits": fits.pop("post_final_layer_norm"), "controls": fits,
        "policy": _policy_metrics(test.beliefs, test.actions, test.rewards, kernels),
        "previous_action_latest_token_controller": lookup,
        "metadata": {
            "checkpoint": str(checkpoint), "report_file": filename, "seed": context.seed,
            "smoke": context.smoke, "device": "cpu", "representation": "post_final_layer_norm",
            "feature_width": train.activations.shape[1], "ridge": RIDGE,
            "model_config": module.config.to_dict(), "sequence_lookback": module.sequence_lookback,
            "sampling_distribution": "process_weighted_rollout", "policy_mode": "stochastic",
            "n_envs_per_split": n_envs, "warmup_per_environment": warmup,
            "rollout_seed_spawn_keys": {key: list(seed.spawn_key) for key, seed in streams.items()},
            "target_alignment": "b_t from info.belief_current, paired with the embedding selecting a_t before env.step(a_t)",
            "actor_inputs": "latest token one-hot (2), previous executed action one-hot (4); all zero at reset",
            "rewards_condition_actor_or_filter": False,
            "coarse_target": "(M1+M2,E,S); exact quotient only in variant 2, aggregation only in variant 3",
            "marginal_features": "Exact next reward for all four actions and next token for every action/token; not joint token/reward predictions",
            "branch_keys": "2 * previous_action + latest_token; reset=8; deployable baseline uses fit centroids with fit global-mean fallback",
            "conditional_metrics": "Evaluation-set branch centroids decompose variance; separate from train-fitted categorical baseline",
            "shuffled_history_control": "Independently permute complete action-token pairs within each full environment history, retaining reset; replay network from empty state and fit/score against ORIGINAL time-indexed targets. Alignment is deliberately broken; not a causal intervention or a null required to have zero R2.",
            "bootstrap": "Whole held-out environment trajectories; fixed probe, 20 smoke or 200 full resamples; not training-seed uncertainty",
            "initialization_control": "Run the same analysis on the actual saved zero-step checkpoint; no surrogate random network is created",
            "interpretation": "Held-out affine accessibility, not causal use. A single training-label permutation is descriptive, not a calibrated significance test.",
        },
    })
    context.results_dir.mkdir(parents=True, exist_ok=True)
    (context.results_dir / filename).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report
