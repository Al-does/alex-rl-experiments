"""Steer the frozen Joint Reward policy toward the moved-reward best actions.

The source policy (30M Joint Reward seed 42) was trained with both factors
paying for code state 2.  Here factor 2 pays for code state 0 instead; the
dynamics, emissions, and actions are unchanged.  At each step the decoded
factor-2 belief selects the myopic best action under the moved reward
(``action_map``); the steering target is a factor-2 point mass on the state in
which the source policy already takes that action (``SOURCE_ACTION_STATE``).
Factor 1 is kept at its decoded belief.  Maps, sites, and the
``h' = h + lambda (target - decoded) E`` update follow
``two_factor_reward_state_REINFORCE_cycle_4/belief_causal_followup.py`` at
revision 9ecb94542a2d0977e1e369040c956ca8d6353cbe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import envs

from analysis.checkpoints import load_module_only
from analysis.rollouts import collect_batched_rollout_data
from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from experiments.two_factor_moved_reward_steering_2026_11.action_map import (
    FACTOR_ACTIONS,
    SOURCE_ACTION_STATE,
    best_factor_action,
    best_joint_action,
    expected_next_reward,
)
from experiments.two_factor_moved_reward_steering_2026_11.task import (
    MOVED_REWARD_STATES,
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.analysis import (
    _target_adapter,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    decode_joint_indices,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import ACTION_PAIRS


HORIZON = 1024
WARMUP = 8
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5)
TARGETS = ("marginal", "joint")
ACTION_TABLE = np.asarray(ACTION_PAIRS, dtype=np.int64)
ANCHOR_STATE = np.zeros(max(FACTOR_ACTIONS) + 1, dtype=np.int64)
for _action, _state in SOURCE_ACTION_STATE.items():
    ANCHOR_STATE[_action] = _state


def affine_fit(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack((features, np.ones(len(features))))
    fitted = np.linalg.lstsq(x, targets, rcond=1e-10)[0]
    return fitted[:-1], fitted[-1]


def r_squared(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = np.sum((prediction - truth) ** 2)
    variance = np.sum((truth - truth.mean(0)) ** 2)
    return float(1 - error / variance)


@dataclass
class BeliefMap:
    decoder: np.ndarray
    bias: np.ndarray
    embedding: np.ndarray
    target: str


def fit_map(activation: np.ndarray, belief: np.ndarray, target: str) -> BeliefMap:
    decoder, bias = affine_fit(activation, belief)
    embedding, _ = affine_fit(belief, activation)
    return BeliefMap(decoder, bias, embedding, target)


def decoded_factor_beliefs(decoded: np.ndarray, target: str) -> np.ndarray:
    """``[n, factor, state]`` factor beliefs implied by a decoder output."""
    if target == "marginal":
        pairs = decoded.reshape(-1, 2, 2)
        return np.concatenate((pairs, 1 - pairs.sum(axis=2, keepdims=True)), axis=2)
    if target == "joint":
        joint = np.column_stack((decoded, 1 - decoded.sum(axis=1))).reshape(-1, 3, 3)
        return np.stack((joint.sum(axis=2), joint.sum(axis=1)), axis=1)
    raise ValueError(f"unknown target: {target}")


def steering_delta(
    activation: np.ndarray, mapping: BeliefMap, reward_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Residual shift and the factor-2 anchor state chosen for each row."""
    decoded = activation @ mapping.decoder + mapping.bias
    factors = decoded_factor_beliefs(decoded, mapping.target)
    desired = best_factor_action(factors[:, 1], reward_state)
    anchor = ANCHOR_STATE[desired]
    point = np.eye(3)[anchor]
    if mapping.target == "marginal":
        target = decoded.copy()
        target[:, 2:] = point[:, :2]
    else:
        joint = np.column_stack((decoded, 1 - decoded.sum(axis=1))).reshape(-1, 3, 3)
        first = joint.sum(axis=2)
        target = (first[:, :, None] * point[:, None, :]).reshape(-1, 9)[:, :8]
    return (target - decoded) @ mapping.embedding, anchor


class ResidualSites:
    """Capture a block's post-MLP residual; optionally shift that residual."""

    def __init__(self, module: torch.nn.Module, reward_state: int):
        self.module = module
        self.reward_state = reward_state
        self.before: dict[int, torch.Tensor] = {}
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.steer_layer: int | None = None
        self.mapping: BeliefMap | None = None
        self.strength = 0.0
        self.null_direction: np.ndarray | None = None

    def shift(self, activation: np.ndarray) -> np.ndarray:
        shift, _ = steering_delta(activation, self.mapping, self.reward_state)
        if self.null_direction is not None:
            shift = np.linalg.norm(shift, axis=1, keepdims=True) * self.null_direction
        return self.strength * shift

    def __enter__(self) -> ResidualSites:
        for layer, block in enumerate(self.module.encoder.blocks):
            def before_hook(_module, inputs, _output, index: int = layer) -> None:
                self.before[index] = inputs[0]

            def after_hook(_module, _inputs, output, index: int = layer):
                activation = self.before[index] + output
                self.activations[index] = activation[:, -1, :]
                if self.steer_layer != index or self.mapping is None:
                    return None
                shift = self.shift(self.activations[index].detach().cpu().numpy())
                return output + torch.as_tensor(
                    shift[:, None, :], dtype=output.dtype, device=output.device
                )

            self.handles.append(block.ln2.register_forward_hook(before_hook))
            self.handles.append(block.mlp.register_forward_hook(after_hook))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def collect_sites(module: torch.nn.Module, *, steps: int, seed: int):
    config = environment_config()
    config["diagnostics"] = {"belief": True, "state": True}
    device = torch.device("cpu")
    n_layers = len(module.encoder.blocks)

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices):
        fresh = initial_state(len(indices))
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    with ResidualSites(module, MOVED_REWARD_STATES[1]) as sites:
        def step_adapter(observations, state, randomness, action_spaces):
            del randomness, action_spaces
            normalized, state_out = module.encode_step(
                torch.as_tensor(observations, dtype=torch.float32), state
            )
            logits = module.action_distribution_inputs(normalized)
            representations = torch.stack(
                [sites.activations[layer] for layer in range(n_layers)] + [normalized],
                dim=1,
            )
            return logits.argmax(dim=-1).cpu().numpy(), state_out, representations.cpu().numpy()

        return collect_batched_rollout_data(
            lambda: HMMEnv(config), step_adapter, _target_adapter,
            n_steps=steps, seed=np.random.SeedSequence([seed, 0]),
            n_envs=8, initial_state=initial_state, reset_state=reset_state,
            warmup=WARMUP,
        )


def _factor_beliefs(infos) -> np.ndarray:
    joint = np.stack([info["belief_current"] for info in infos]).reshape(-1, 3, 3)
    return np.stack((joint.sum(axis=2), joint.sum(axis=1)), axis=1)


@torch.inference_mode()
def evaluate(
    module: torch.nn.Module, mapping: BeliefMap | None, *,
    site: int, strength: float, episodes: int, seed: int,
    policy: str = "agent", matched_null: bool = False,
) -> dict[str, object]:
    """Complete-episode rewards under the moved reward for one policy variant.

    ``policy``: ``agent`` (optionally steered), ``random`` (uniform joint
    actions), or ``belief_map`` (myopic best action from the exact belief).
    """
    if episodes < 2:
        raise ValueError("at least two episodes are required for uncertainty estimates")
    reward_states = MOVED_REWARD_STATES
    rng = np.random.default_rng(seed)
    episode_seeds = rng.integers(0, 2**31, size=episodes)
    action_seeds = rng.integers(0, 2**31, size=episodes)
    config = environment_config()
    config["randomize_first_episode_length"] = False
    config["diagnostics"] = {"state": True, "belief": True}
    null_direction = None
    if matched_null:
        null_direction = np.random.default_rng(seed + 9999).standard_normal(
            mapping.decoder.shape[0]
        )
        null_direction -= (null_direction @ mapping.decoder) @ np.linalg.pinv(
            mapping.decoder, rcond=1e-10
        )
        null_direction /= np.linalg.norm(null_direction)
    totals = np.zeros((episodes, 2))
    source_factor_2 = np.zeros(episodes)
    expected_next = np.zeros((episodes, 2))
    agrees_with_map = np.zeros((episodes, 2))
    changed = np.zeros(2)
    n_layers = len(module.encoder.blocks)
    for offset in range(0, episodes, 16):
        count = min(16, episodes - offset)
        envs = [HMMEnv(config) for _ in range(count)]
        try:
            observations, infos = zip(*(
                env.reset(seed=int(value))
                for env, value in zip(envs, episode_seeds[offset:offset + count])
            ))
            observations = np.stack(observations)
            infos = list(infos)
            generators = [np.random.default_rng(int(v)) for v in action_seeds[offset:offset + count]]
            state = _initial_state(module, count, torch.device("cpu"))
            rows = slice(offset, offset + count)
            with ResidualSites(module, reward_states[1]) as sites:
                for _ in range(HORIZON):
                    tensor = torch.as_tensor(observations, dtype=torch.float32)
                    sites.steer_layer = None
                    normalized_base, base_state = module.encode_step(tensor, state)
                    base_actions = module.action_distribution_inputs(
                        normalized_base
                    ).argmax(dim=-1).cpu().numpy()
                    beliefs = _factor_beliefs(infos)
                    oracle = best_joint_action(beliefs, reward_states)
                    if policy == "random":
                        actions = np.asarray([g.integers(9) for g in generators])
                    elif policy == "belief_map":
                        actions = oracle
                    elif mapping is None or strength == 0:
                        actions, state = base_actions, base_state
                    elif site == n_layers:
                        normalized, state = module.encode_step(tensor, state)
                        sites.mapping, sites.strength = mapping, strength
                        sites.null_direction = null_direction
                        shift = sites.shift(normalized.cpu().numpy())
                        actions = module.action_distribution_inputs(
                            normalized + torch.as_tensor(shift, dtype=normalized.dtype)
                        ).argmax(dim=-1).cpu().numpy()
                    else:
                        sites.steer_layer = site
                        sites.mapping, sites.strength = mapping, strength
                        sites.null_direction = null_direction
                        normalized, state = module.encode_step(tensor, state)
                        actions = module.action_distribution_inputs(
                            normalized
                        ).argmax(dim=-1).cpu().numpy()
                    if policy != "agent":
                        state = base_state
                    executed = ACTION_TABLE[actions]
                    changed += (executed != ACTION_TABLE[base_actions]).sum(axis=0)
                    agrees_with_map[rows] += executed == ACTION_TABLE[oracle]
                    for factor in range(2):
                        table = expected_next_reward(beliefs[:, factor], reward_states[factor])
                        column = np.searchsorted(FACTOR_ACTIONS, executed[:, factor])
                        expected_next[rows, factor] += table[np.arange(count), column]
                    latent = decode_joint_indices(
                        np.asarray([int(info["state_current"]) for info in infos])
                    )
                    totals[rows] += latent == np.asarray(reward_states)
                    source_factor_2[rows] += latent[:, 1] == 2
                    next_observations = []
                    for index, env in enumerate(envs):
                        next_obs, _, terminated, truncated, info = env.step(int(actions[index]))
                        if (terminated or truncated) and int(info["decision_step"]) != HORIZON:
                            raise ValueError("evaluation requires full 1024-step episodes")
                        next_observations.append(next_obs)
                        infos[index] = info
                    observations = np.stack(next_observations)
        finally:
            for env in envs:
                env.close()

    def summary(values: np.ndarray) -> tuple[list[float], list[float]]:
        values = values / HORIZON
        return (values.mean(axis=0).tolist(),
                (values.std(axis=0, ddof=1) / np.sqrt(episodes)).tolist())

    reward, reward_se = summary(totals)
    nxt, nxt_se = summary(expected_next)
    agree, _ = summary(agrees_with_map)
    old, old_se = summary(source_factor_2[:, None])
    return {
        "factor_reward": reward, "factor_reward_se": reward_se,
        "expected_next_step_reward": nxt, "expected_next_step_reward_se": nxt_se,
        "factor_action_agrees_with_best_action_map": agree,
        "factor_2_source_reward_state_occupancy": old[0],
        "factor_2_source_reward_state_occupancy_se": old_se[0],
        "factor_action_change_fraction": (changed / (episodes * HORIZON)).tolist(),
    }


def learned_action_map(data) -> dict[str, object]:
    """Source-policy factor actions tabulated by the exact belief's argmax state."""
    beliefs = data.targets["factor_belief"]
    actions = ACTION_TABLE[np.asarray(data.actions).reshape(-1)]
    out = {}
    for factor in range(2):
        state = beliefs[:, factor].argmax(axis=1)
        table = np.zeros((3, 3))
        for s, a in zip(state, actions[:, factor]):
            table[s, a] += 1
        out[f"factor_{factor + 1}"] = (table / table.sum(axis=1, keepdims=True)).tolist()
    return {"rows_argmax_state_cols_action_noop_x_y": out}


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def run(checkpoint: Path, output: Path, *, steps: int, tune_episodes: int,
        confirm_episodes: int, seed: int, arrays: Path | None) -> None:
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    module = load_module_only(checkpoint).to("cpu").eval()
    train = collect_sites(module, steps=steps, seed=seed)
    test = collect_sites(module, steps=steps, seed=seed + 1)
    n_layers = len(module.encoder.blocks)
    targets = {
        "marginal": (train.targets["factor_belief"][:, :, :2].reshape(-1, 4),
                     test.targets["factor_belief"][:, :, :2].reshape(-1, 4)),
        "joint": (train.targets["joint_belief"][:, :8], test.targets["joint_belief"][:, :8]),
    }
    maps: dict[tuple[int, str], BeliefMap] = {}
    probes: dict[str, dict[str, object]] = {}
    for site in range(n_layers + 1):
        name = f"block_{site + 1}_pre_final_norm" if site < n_layers else "post_final_norm"
        probes[name] = {}
        for target, (y_train, y_test) in targets.items():
            mapping = fit_map(train.representations[:, site].astype(np.float64),
                              y_train.astype(np.float64), target)
            maps[(site, target)] = mapping
            predicted = test.representations[:, site] @ mapping.decoder + mapping.bias
            probes[name][target] = {"heldout_r2": r_squared(predicted, y_test)}
    if arrays is not None:
        final = maps[(n_layers - 1, "joint")]
        decoded = test.representations[:, n_layers - 1] @ final.decoder + final.bias
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays, factor_belief=test.targets["factor_belief"],
            decoded_factor_belief=decoded_factor_beliefs(decoded, "joint"),
            actions=np.asarray(test.actions).reshape(-1),
        )

    final_site = n_layers - 1
    tune_seed, confirm_seed = seed + 1000, seed + 2000
    tuning = {
        "intact": evaluate(module, None, site=final_site, strength=0,
                           episodes=tune_episodes, seed=tune_seed),
        "random": evaluate(module, None, site=final_site, strength=0,
                           episodes=tune_episodes, seed=tune_seed, policy="random"),
    }
    sweeps, selected = {}, {}
    for target in TARGETS:
        sweep = {str(s): evaluate(module, maps[(final_site, target)], site=final_site,
                                  strength=s, episodes=tune_episodes, seed=tune_seed)
                 for s in STRENGTHS}
        best = max(sweep[str(s)]["factor_reward"][1] for s in STRENGTHS)
        selected[target] = min(s for s in STRENGTHS if sweep[str(s)]["factor_reward"][1] == best)
        sweeps[target] = sweep
    comparisons = {}
    for site in range(n_layers + 1):
        for target in TARGETS:
            name = f"block_{site + 1}_{target}" if site < n_layers else f"post_norm_{target}"
            comparisons[name] = {"strength": selected[target], **evaluate(
                module, maps[(site, target)], site=site, strength=selected[target],
                episodes=confirm_episodes, seed=confirm_seed)}
    confirmation = {
        "intact": evaluate(module, None, site=final_site, strength=0,
                           episodes=confirm_episodes, seed=confirm_seed),
        "random": evaluate(module, None, site=final_site, strength=0,
                           episodes=confirm_episodes, seed=confirm_seed, policy="random"),
        "exact_belief_best_action_map": evaluate(
            module, None, site=final_site, strength=0, episodes=confirm_episodes,
            seed=confirm_seed, policy="belief_map"),
        "matched_null": {target: evaluate(
            module, maps[(final_site, target)], site=final_site, strength=selected[target],
            episodes=confirm_episodes, seed=confirm_seed, matched_null=True)
            for target in TARGETS},
        "comparisons": comparisons,
    }
    here = Path(__file__).resolve()
    report = {
        "checkpoint": str(checkpoint),
        "module_state_sha256": hashlib.sha256(
            (checkpoint / "module_state.pkl").read_bytes()).hexdigest(),
        "experiment_revision": _git_revision(here.parent),
        "harness_revision": _git_revision(Path(envs.__file__).resolve().parent),
        "reward_states_code": list(MOVED_REWARD_STATES),
        "source_reward_states_code": [2, 2],
        "fit_seed": seed, "test_seed": seed + 1, "fit_test_steps_each": steps,
        "warmup_steps": WARMUP, "tune_seed": tune_seed, "tune_episodes": tune_episodes,
        "confirm_seed": confirm_seed, "confirm_episodes": confirm_episodes,
        "episode_horizon": HORIZON,
        "probe_sites": probes,
        "learned_action_map_fit_rollout": learned_action_map(train),
        "strength_tuning": {**tuning, "grid_by_target": sweeps, "selected_strength": selected,
                            "selection": "smallest grid strength maximizing tuning factor-2 reward"},
        "confirmation": confirmation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrays", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--tune-episodes", type=int, default=16)
    parser.add_argument("--confirm-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.checkpoint, args.output, steps=args.steps, tune_episodes=args.tune_episodes,
        confirm_episodes=args.confirm_episodes, seed=args.seed, arrays=args.arrays)


if __name__ == "__main__":
    main()
