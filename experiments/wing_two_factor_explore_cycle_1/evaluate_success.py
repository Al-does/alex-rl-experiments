from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch

from envs.wing.model import controlled_kernels
from experiments.wing_two_factor_explore_cycle_1.control_data import collect_control_data, replay_beliefs


STUDIES = {
    "token_guess": ("wing_token_guess_cycle_1/ppo", "20260908T000438Z-b9d3cfaa"),
    "reward_both": ("wing_two_factor_explore_cycle_2/reward_both_state_0", "20260908T001136Z-9064aabc"),
    "reward_factor_1": ("wing_two_factor_explore_cycle_2/reward_factor_1_state_0", "20260908T001436Z-51c1f6cb"),
}


def episode_statistics(values, episode_ids, horizon):
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(episode_ids)
    if values.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("success values and episode IDs must be finite aligned vectors")
    if ((values < 0) | (values > 1 + 1e-12)).any():
        raise ValueError("success values must be probabilities")
    groups = np.unique(ids)
    if len(groups) < 2:
        raise ValueError("uncertainty requires at least two independent episodes")
    means = []
    for group in groups:
        rows = values[ids == group]
        if len(rows) != horizon:
            raise ValueError("success evaluation requires complete equal-length episodes")
        means.append(float(rows.mean()))
    means = np.asarray(means)
    mean = float(means.mean())
    standard_error = float(means.std(ddof=1) / np.sqrt(len(means)))
    half_width = float(t.ppf(0.975, len(means) - 1) * standard_error)
    return {
        "mean": mean,
        "ci95": [mean - half_width, mean + half_width],
        "standard_error": standard_error,
        "n_episodes": len(groups),
        "n_decisions": len(values),
        "horizon": horizon,
        "episode_means": means.tolist(),
        "uncertainty": "Student-t interval over independent complete-episode conditional-expected-success means; one fixed trained model",
    }


def policy_success_values(data, *, parameters, condition, reward_state=0):
    beliefs, marginal_ntp = replay_beliefs(data, **parameters)
    if not np.allclose(beliefs, data.beliefs, atol=1e-10, rtol=0):
        raise ValueError("replayed decision beliefs disagree with environment diagnostics")
    probabilities = np.asarray(data.policy_probabilities, dtype=np.float64)
    expected_width = 4 if condition == "token_guess" else 9
    if probabilities.shape != (len(beliefs), expected_width) or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("policy probabilities must match the task action space")
    if condition == "token_guess":
        if parameters["strength"] is not None:
            raise ValueError("token guessing requires passive dynamics")
        joint_ntp = (marginal_ntp[:, 0, :, None] * marginal_ntp[:, 1, None, :]).reshape(-1, 4)
        return (probabilities * joint_ntp).sum(axis=1)
    if condition not in ("reward_both", "reward_factor_1") or reward_state not in (0, 1, 2):
        raise ValueError("unknown reward condition or state")
    if parameters["strength"] is None:
        raise ValueError("reward occupancy requires action-conditioned dynamics")
    transitions = controlled_kernels(**parameters).sum(axis=1)
    factor_values = np.einsum("nfs,as->nfa", beliefs, transitions[:, :, reward_state])
    first = np.repeat(factor_values[:, 0], 3, axis=1)
    values = (first + np.tile(factor_values[:, 1], (1, 3))) / 2 if condition == "reward_both" else first
    return (probabilities * values).sum(axis=1)


def evaluate_module_success(module, *, env_config, condition, n_episodes=128, horizon=1024, n_envs=8, seed=20260909):
    if n_episodes < 2 or n_envs < 1 or n_episodes % n_envs:
        raise ValueError("episode count must be at least two and divisible by n_envs")
    config = deepcopy(env_config)
    config["episode_length"] = horizon
    config["randomize_first_episode_length"] = False
    parameters = dict(config["model"]["kwargs"]["factors"][0]["kwargs"])
    parameters["strength"] = config["task"].get("kwargs", {}).get("strength")
    reward_state = config["task"].get("kwargs", {}).get("reward_state", 0)
    data = collect_control_data(
        module, env_config=config, n_steps=n_episodes * horizon, seed=seed,
        device=torch.device("cpu"), n_envs=n_envs, warmup=0,
    )
    values = policy_success_values(data, parameters=parameters, condition=condition, reward_state=reward_state)
    result = episode_statistics(values, data.episode_ids, horizon)
    if result["n_episodes"] != n_episodes:
        raise AssertionError("collector did not produce the requested complete episodes")
    result.update({
        "metric": "joint_token_accuracy" if condition == "token_guess" else "mean_rewarded_factor_arrival_occupancy",
        "estimator": "mean exact conditional success probability under the learned stochastic policy on its own on-policy histories",
        "condition": condition,
        "seed": seed,
        "n_envs": n_envs,
        "warmup": 0,
        "environment_config": config,
        "policy_mode": "learned_stochastic_with_checkpoint_temperature",
        "scope": "evaluation-sample uncertainty, not variation across training seeds",
    })
    return result


def resolve_checkpoint(repo, leaf, run_id):
    leaf_dir = repo / "experiments" / leaf
    source = leaf_dir / "results" / run_id
    summary = json.loads((source / "tune_summary.json").read_text())
    checkpoint = Path(summary["trials"][0]["checkpoint"])
    parts = checkpoint.parts
    if "artifacts" not in parts:
        raise ValueError("recorded checkpoint has no artifact-root-relative path")
    local = leaf_dir.joinpath(*parts[parts.index("artifacts"):])
    module_path = local / "learner_group" / "learner" / "rl_module" / "default_policy"
    if not module_path.is_dir():
        raise FileNotFoundError(f"Download the recorded final module subtree first: {module_path}")
    return module_path, source, summary


def bayes_references(conditions, seed=20260910):
    from experiments.wing_two_factor_explore_cycle_1.success_benchmarks import (
        controlled_bayes_refinements, controlled_belief_policy_reward, token_guess_bayes_accuracy,
    )

    passive_row = conditions["token_guess"]
    passive_parameters = passive_row["environment_config"]["model"]["kwargs"]["factors"][0]["kwargs"]
    passive = token_guess_bayes_accuracy(**passive_parameters, horizon=passive_row["horizon"], seed=seed)
    analytic = passive["analytic_cross_check"]
    token_reference = {
        "value": analytic["fraction"] if analytic else passive["mean"],
        "kind": "exact_optimal_accuracy" if analytic else "monte_carlo_optimal_accuracy",
        "label": "Bayes maximum (exact dominant-token rule)" if analytic else "Bayes-optimal accuracy (MC estimate)",
        "ci95": None if analytic else passive["ci95"],
        "details": passive,
    }
    controlled_row = conditions["reward_both"]
    config = controlled_row["environment_config"]
    parameters = dict(config["model"]["kwargs"]["factors"][0]["kwargs"])
    parameters.update({"strength": config["task"]["kwargs"]["strength"], "reward_state": config["task"]["kwargs"]["reward_state"], "horizon": controlled_row["horizon"]})
    other = conditions["reward_factor_1"]["environment_config"]
    if other["model"] != config["model"] or other["task"]["kwargs"]["strength"] != parameters["strength"] or other["task"]["kwargs"]["reward_state"] != parameters["reward_state"] or conditions["reward_factor_1"]["horizon"] != parameters["horizon"]:
        raise ValueError("a shared controlled benchmark requires identical factor dynamics, reward state, and horizon")
    bounds = controlled_bayes_refinements(**parameters)
    attainable = controlled_belief_policy_reward(**parameters, seed=seed + 1)
    controlled_reference = {
        "value": bounds["upper_bound_fraction"],
        "kind": "numerical_upper_bound",
        "label": "Bayes planning upper bound (numerical)",
        "ci95": None,
        "details": {"grid_refinements": bounds, "attainable_policy": attainable},
    }
    return {"token_guess": token_reference, "reward_both": controlled_reference, "reward_factor_1": controlled_reference}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate complete-episode task success for the three completed Wing runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")
    from ray.rllib.core.rl_module.rl_module import RLModule
    from experiments.wing_token_guess_cycle_1.process import environment_config as passive_config
    from experiments.wing_two_factor_explore_cycle_2.process import environment_config as controlled_config

    repo = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1)
    conditions = {}
    for condition, (leaf, run_id) in STUDIES.items():
        module_path, source, summary = resolve_checkpoint(repo, leaf, run_id)
        module = RLModule.from_checkpoint(str(module_path))
        config = passive_config() if condition == "token_guess" else controlled_config(condition)
        result = evaluate_module_success(module, env_config=config, condition=condition, n_episodes=args.episodes, seed=args.seed)
        result.update({
            "checkpoint": str(module_path.relative_to(repo)),
            "source_run": str(source.relative_to(repo)),
            "agent_steps": int(summary["trials"][0]["metrics"]["env_runners/num_env_steps_sampled_lifetime"]),
            "checkpoint_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(module_path.iterdir()) if path.is_file()},
        })
        conditions[condition] = result
        print(f"{condition}: {100 * result['mean']:.4f}% [{100 * result['ci95'][0]:.4f}, {100 * result['ci95'][1]:.4f}]", flush=True)
        del module
    references = bayes_references(conditions, seed=args.seed + 1)
    for condition, reference in references.items():
        conditions[condition]["bayes_reference"] = reference
        print(f"{condition}: {reference['label']} = {100 * reference['value']:.6f}%", flush=True)
    payload = {
        "schema_version": 1,
        "conditions": conditions,
        "metadata": {
            "seed": args.seed,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "success_units": "probability fractions; multiply by 100 for percentages",
            "evaluation": "independent full 1024-step episodes, including all reset transients; no activation intervention",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
