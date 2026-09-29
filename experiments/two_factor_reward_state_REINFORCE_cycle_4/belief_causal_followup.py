"""Held-out factor-belief interventions at cached transformer residual sites."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from analysis.rollouts import collect_batched_rollout_data
from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from experiments.two_factor_reward_state_PPO_cycle_2.process import (
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.analysis import (
    _target_adapter,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import ACTION_PAIRS
from experiments.two_factor_reward_state_REINFORCE_cycle_4.belief_intervention import (
    affine_fit,
    r_squared,
)


@dataclass
class BeliefMap:
    decoder: np.ndarray
    bias: np.ndarray
    embedding: np.ndarray
    mean: np.ndarray
    target: str


def fit_map(activation: np.ndarray, belief: np.ndarray, target: str) -> BeliefMap:
    decoder, bias = affine_fit(activation, belief)
    embedding, _ = affine_fit(belief, activation)
    return BeliefMap(decoder, bias, embedding, belief.mean(axis=0), target)


def opposite_delta(activation: np.ndarray, mapping: BeliefMap) -> np.ndarray:
    decoded = activation @ mapping.decoder + mapping.bias
    if mapping.target == "marginal":
        target = decoded.copy()
        target[:, 2] = (decoded[:, 3] > decoded[:, 2]).astype(np.float64)
        target[:, 3] = 1 - target[:, 2]
        difference = target - decoded
    elif mapping.target == "joint":
        joint = np.column_stack((decoded, 1 - decoded.sum(axis=1))).reshape(-1, 3, 3)
        first = joint.sum(axis=2)
        second = joint.sum(axis=1)
        opposite = np.zeros_like(second)
        opposite[:, 0] = (second[:, 1] > second[:, 0]).astype(np.float64)
        opposite[:, 1] = 1 - opposite[:, 0]
        difference = (first[:, :, None] * opposite[:, None, :] - joint).reshape(-1, 9)[:, :8]
    else:
        raise ValueError(f"unknown target: {mapping.target}")
    return difference @ mapping.embedding


class ResidualSites:
    """Capture a block's residual after its MLP; optionally shift that residual."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.before: dict[int, torch.Tensor] = {}
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.steer_layer: int | None = None
        self.mapping: BeliefMap | None = None
        self.strength = 0.0
        self.null_direction: np.ndarray | None = None

    def __enter__(self) -> ResidualSites:
        for layer, block in enumerate(self.module.encoder.blocks):
            def before_hook(
                _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...],
                _output: torch.Tensor, index: int = layer,
            ) -> None:
                self.before[index] = inputs[0]

            def after_hook(
                _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor, index: int = layer,
            ) -> torch.Tensor | None:
                activation = self.before[index] + output
                self.activations[index] = activation[:, -1, :]
                if self.steer_layer != index or self.mapping is None:
                    return None
                shift = opposite_delta(
                    self.activations[index].detach().cpu().numpy(), self.mapping
                )
                if self.null_direction is not None:
                    shift = np.linalg.norm(shift, axis=1, keepdims=True) * self.null_direction
                shift *= self.strength
                return output + torch.as_tensor(
                    shift[:, None, :], dtype=output.dtype, device=output.device
                )

            self.handles.append(block.ln2.register_forward_hook(before_hook))
            self.handles.append(block.mlp.register_forward_hook(after_hook))
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def collect_sites(module: torch.nn.Module, *, steps: int, seed: int):
    config = environment_config("reward_both")
    config["diagnostics"] = {"belief": True, "state": True}
    device = torch.device("cpu")
    n_layers = len(module.encoder.blocks)

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state: dict[str, torch.Tensor], indices: np.ndarray):
        fresh = initial_state(len(indices))
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    with ResidualSites(module) as sites:
        def step_adapter(observations, state, randomness, action_spaces):
            del randomness, action_spaces
            normalized, state_out = module.encode_step(
                torch.as_tensor(observations, dtype=torch.float32), state
            )
            logits = module.action_distribution_inputs(normalized)
            representations = torch.stack(
                [sites.activations[layer] for layer in range(n_layers)]
                + [normalized], dim=1
            )
            return logits.argmax(dim=-1).cpu().numpy(), state_out, representations.cpu().numpy()

        return collect_batched_rollout_data(
            lambda: HMMEnv(config), step_adapter, _target_adapter,
            n_steps=steps, seed=np.random.SeedSequence([seed, 0]),
            n_envs=8, initial_state=initial_state, reset_state=reset_state,
            warmup=8,
        )


@torch.inference_mode()
def evaluate(
    module: torch.nn.Module, mapping: BeliefMap, *,
    site: int, strength: float, episodes: int, seed: int, random_actions: bool = False,
    matched_null: bool = False,
) -> dict[str, object]:
    if episodes < 2:
        raise ValueError("at least two episodes are required for uncertainty estimates")
    rng = np.random.default_rng(seed)
    episode_seeds = rng.integers(0, 2**31, size=episodes)
    action_seeds = rng.integers(0, 2**31, size=episodes)
    config = environment_config("reward_both")
    config["randomize_first_episode_length"] = False
    config["diagnostics"] = {"state": True}
    null_direction = None
    if matched_null:
        null_direction = np.random.default_rng(seed + 9999).standard_normal(
            mapping.decoder.shape[0]
        )
        null_direction -= (
            null_direction @ mapping.decoder
        ) @ np.linalg.pinv(mapping.decoder, rcond=1e-10)
        null_direction /= np.linalg.norm(null_direction)
    totals = np.zeros((episodes, 2))
    changed = 0
    changes_by_factor = np.zeros(2)
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
            generators = [
                np.random.default_rng(int(value))
                for value in action_seeds[offset:offset + count]
            ]
            state = _initial_state(module, count, torch.device("cpu"))
            with ResidualSites(module) as sites:
                for _ in range(1024):
                    tensor = torch.as_tensor(observations, dtype=torch.float32)
                    sites.steer_layer = None
                    normalized_base, _ = module.encode_step(tensor, state)
                    base_actions = module.action_distribution_inputs(
                        normalized_base
                    ).argmax(dim=-1).cpu().numpy()
                    if random_actions:
                        actions = np.asarray([g.integers(9) for g in generators])
                    elif site == n_layers:
                        normalized, state = module.encode_step(tensor, state)
                        shift = strength * opposite_delta(
                            normalized.cpu().numpy(), mapping
                        )
                        if null_direction is not None:
                            shift = np.linalg.norm(shift, axis=1, keepdims=True) * null_direction
                        logits = module.action_distribution_inputs(
                            normalized + torch.as_tensor(shift, dtype=normalized.dtype)
                        )
                        actions = logits.argmax(dim=-1).cpu().numpy()
                    else:
                        sites.steer_layer = site
                        sites.mapping = mapping
                        sites.strength = strength
                        sites.null_direction = null_direction
                        normalized, state = module.encode_step(tensor, state)
                        logits = module.action_distribution_inputs(normalized)
                        actions = logits.argmax(dim=-1).cpu().numpy()
                    changed += int(np.count_nonzero(actions != base_actions))
                    for factor in range(2):
                        changes_by_factor[factor] += sum(
                            ACTION_PAIRS[int(action)][factor]
                            != ACTION_PAIRS[int(base)][factor]
                            for action, base in zip(actions, base_actions)
                        )
                    next_observations = []
                    for index, env in enumerate(envs):
                        latent = int(infos[index]["state_current"])
                        totals[offset + index] += (
                            latent // 3 == 2, latent % 3 == 2
                        )
                        next_obs, _, terminated, truncated, info = env.step(int(actions[index]))
                        if (terminated or truncated) and int(info["decision_step"]) != 1024:
                            raise ValueError("evaluation requires full 1024-step episodes")
                        next_observations.append(next_obs)
                        infos[index] = info
                    observations = np.stack(next_observations)
        finally:
            for env in envs:
                env.close()
    occupied = totals / 1024
    return {
        "factor_occupancy": occupied.mean(axis=0).tolist(),
        "factor_occupancy_se": (occupied.std(axis=0, ddof=1) / np.sqrt(episodes)).tolist(),
        "factor_action_change_fraction": (changes_by_factor / (episodes * 1024)).tolist(),
        "joint_action_change_fraction": changed / (episodes * 1024),
    }


def run(checkpoint: Path, output: Path, *, steps: int, tune_episodes: int,
        confirm_episodes: int, seed: int) -> None:
    torch.set_num_threads(1)
    module = load_module_only(checkpoint).to("cpu").eval()
    train = collect_sites(module, steps=steps, seed=seed)
    test = collect_sites(module, steps=steps, seed=seed + 1)
    n_layers = len(module.encoder.blocks)
    targets = {
        "marginal": (
            train.targets["factor_belief"][:, :, :2].reshape(-1, 4),
            test.targets["factor_belief"][:, :, :2].reshape(-1, 4),
        ),
        "joint": (train.targets["joint_belief"][:, :8],
                  test.targets["joint_belief"][:, :8]),
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
            probes[name][target] = {
                "heldout_r2": r_squared(predicted, y_test),
                "heldout_mse": float(np.mean((predicted - y_test) ** 2)),
                "target_variance": float(np.mean((y_test - y_test.mean(axis=0)) ** 2)),
            }

    strengths = (0.0, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
    final_site = n_layers - 1
    tune_seed = seed + 1000
    confirm_seed = seed + 2000
    baseline = evaluate(module, maps[(final_site, "marginal")], site=final_site,
                        strength=0, episodes=tune_episodes, seed=tune_seed)
    random = evaluate(module, maps[(final_site, "marginal")], site=final_site,
                      strength=0, episodes=tune_episodes, seed=tune_seed,
                      random_actions=True)
    sweeps: dict[str, object] = {}
    selected: dict[str, float] = {}
    for target in ("marginal", "joint"):
        sweep: dict[str, object] = {}
        for strength in strengths:
            sweep[str(strength)] = evaluate(
                module, maps[(final_site, target)], site=final_site,
                strength=strength, episodes=tune_episodes, seed=tune_seed,
            )
        eligible = [
            strength for strength in strengths
            if sweep[str(strength)]["factor_occupancy"][1]
            <= random["factor_occupancy"][1]
        ]
        selected[target] = min(eligible) if eligible else min(
            strengths, key=lambda strength: sweep[str(strength)]["factor_occupancy"][1]
        )
        sweeps[target] = sweep
    comparisons: dict[str, object] = {}
    for site in (0, 1, 2, final_site, n_layers):
        for target in ("marginal", "joint"):
            name = f"block_{site + 1}_{target}" if site < n_layers else f"post_norm_{target}"
            result = evaluate(
                module, maps[(site, target)], site=site, strength=selected[target],
                episodes=confirm_episodes, seed=confirm_seed,
            )
            comparisons[name] = {"strength": selected[target], **result}
    null_controls = {
        target: evaluate(
            module, maps[(final_site, target)], site=final_site,
            strength=selected[target], episodes=confirm_episodes,
            seed=confirm_seed, matched_null=True,
        )
        for target in ("marginal", "joint")
    }
    confirmation_intact = evaluate(
        module, maps[(final_site, "marginal")], site=final_site,
        strength=0, episodes=confirm_episodes, seed=confirm_seed,
    )
    confirmation_random = evaluate(
        module, maps[(final_site, "marginal")], site=final_site,
        strength=0, episodes=confirm_episodes, seed=confirm_seed,
        random_actions=True,
    )
    report = {
        "checkpoint": str(checkpoint),
        "module_state_sha256": hashlib.sha256((checkpoint / "module_state.pkl").read_bytes()).hexdigest(),
        "fit_seed": seed, "test_seed": seed + 1,
        "fit_test_steps_each": steps, "warmup_steps": 8,
        "tune_seed": tune_seed, "tune_episodes": tune_episodes,
        "confirm_seed": confirm_seed, "confirm_episodes": confirm_episodes,
        "episode_horizon": 1024,
        "probe_sites": probes,
        "strength_tuning": {"intact": baseline, "random": random,
                            "grid_by_target": sweeps, "selected_strength": selected,
                            "selection": "smallest grid strength with factor-2 occupancy at or below random; otherwise lowest factor-2 occupancy"},
        "confirmation": {"intact": confirmation_intact,
                         "random": confirmation_random,
                         "matched_null": null_controls,
                         "comparisons": comparisons},
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--tune-episodes", type=int, default=16)
    parser.add_argument("--confirm-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.checkpoint, args.output, steps=args.steps,
        tune_episodes=args.tune_episodes, confirm_episodes=args.confirm_episodes,
        seed=args.seed)


if __name__ == "__main__":
    main()
