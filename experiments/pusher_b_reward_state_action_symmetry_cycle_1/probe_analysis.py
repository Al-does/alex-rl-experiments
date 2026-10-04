"""Post-hoc transducer belief probes for Pusher-B action-symmetry checkpoints.

The Bayes target is the controlled (action-conditioned) filter of the
delay-one edge-emitting Pusher-B process. Row ``t`` of a rollout is the
decision point whose observation carries token ``y_{t-1}`` and executed action
``a_{t-1}``. Its target is

    s_t = normalize(s_{t-1} @ K_{a_{t-1}}[y_{t-1}])     (source posterior)
    b_t = s_t @ T_{a_t}                                  (current-state belief)

where ``K_a[y]`` are the edge kernels selected by the action actually executed
on that edge (the reset edge uses the uncontrolled kernels) and ``T_a`` is the
executed transition matrix of the current step. ``b_t`` matches the
environment's own ``belief_current`` diagnostic, which is asserted exactly.
Source-belief filtering uses ``analysis.probes.filter_operator_histories`` over
complete episodes; warmup rows are dropped only after filtering.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from analysis.checkpoints import load_module_only
from analysis.probes import fit_affine_probe, probe_predict, r2_score
from analysis.probes.transducer import filter_operator_histories
from analysis.rollouts import collect_batched_rollout_data
from envs.hmm import HMMEnv
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from experiments.pusher_b_reward_state_action_symmetry_cycle_1.process import (
    environment_config,
)

PROBE_RIDGE = 1e-6
N_ENVS = 32
_STREAM_KEYS = {"probe_train": (400,), "probe_test": (401,)}
_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "tokens": True,
    "transitions": True,
}


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    beliefs: np.ndarray
    diagnostic_beliefs: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    env_indices: np.ndarray
    episode_steps: np.ndarray
    episode_ids: np.ndarray


def _initial_state(module: Any, batch_size: int, device: torch.device):
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        .to(device)
        for key, value in module.get_initial_state().items()
    }


def make_environment_factory(
    preset: str,
    variant: int,
    reward_state: str,
):
    config = environment_config(preset, variant, reward_state)
    config["diagnostics"] = dict(_DIAGNOSTICS)

    def factory() -> HMMEnv:
        return HMMEnv(dict(config))

    return factory


def transducer_targets(
    initial_belief: np.ndarray,
    source_operators: np.ndarray,
    executed_transitions: np.ndarray,
    groups: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    """Filter source beliefs over full histories, then predict one edge."""

    source = filter_operator_histories(
        initial_belief,
        source_operators,
        groups,
        steps,
    )
    beliefs = np.einsum("ni,nij->nj", source, executed_transitions)
    return beliefs / beliefs.sum(axis=1, keepdims=True)


@torch.no_grad()
def collect_probe_data(
    module: Any,
    env_factory,
    *,
    n_steps: int,
    seed: int,
    n_envs: int = N_ENVS,
    device: str | torch.device = "cpu",
    warmup: int = 0,
    policy_mode: str = "policy",
) -> ProbeData:
    if policy_mode not in {"policy", "greedy"}:
        raise ValueError(f"unsupported policy mode {policy_mode!r}")
    device = torch.device(device)
    module = module.to(device).eval()
    if not module.is_stateful() or not module.heads.is_discrete:
        raise ValueError("Pusher-B probing expects a stateful discrete module")

    reference = env_factory()
    try:
        initial_belief = np.asarray(
            reference.model.initial_distribution,
            dtype=np.float64,
        )
        reset_edges = np.asarray(
            reference.model.edge_transition_matrices,
            dtype=np.float64,
        )
        reset_transition = np.asarray(
            reference.model.transition_matrix,
            dtype=np.float64,
        )
        if reference.config.delay != 1:
            raise ValueError("adapter assumes the delay-one Pusher-B process")
    finally:
        reference.close()
    n_states = len(initial_belief)
    pending_edges = np.repeat(reset_edges[None], n_envs, axis=0)
    episode_ids = np.arange(n_envs, dtype=np.int64)
    next_episode_id = n_envs
    first_call = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_sequence_to_int(np.random.SeedSequence(seed)))

    def initial_state(batch_size: int):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices: np.ndarray):
        fresh = _initial_state(module, len(indices), device)
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    def step_adapter(observations, state, randomness, action_spaces):
        del randomness, action_spaces
        embedding, state = module.encode_step(
            torch.from_numpy(np.asarray(observations)).float().to(device),
            state,
        )
        logits = module.action_distribution_inputs(embedding)
        if policy_mode == "greedy":
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.multinomial(
                torch.softmax(logits, dim=-1).cpu(),
                1,
                generator=generator,
            ).squeeze(-1)
        return actions.cpu().numpy(), state, embedding.cpu().numpy()

    def target_adapter(observations, infos, episode_steps):
        nonlocal next_episode_id, first_call
        del observations
        source_operators = np.empty((n_envs, n_states, n_states))
        transitions = np.empty((n_envs, n_states, n_states))
        for index, (info, step) in enumerate(zip(infos, episode_steps)):
            if step == 0:
                source_operators[index] = np.eye(n_states)
                transitions[index] = reset_transition
                pending_edges[index] = reset_edges
                if not first_call:
                    episode_ids[index] = next_episode_id
                    next_episode_id += 1
            else:
                token = info["visible_token_current"]
                if token is None:
                    raise AssertionError("delay-one row without visible token")
                source_operators[index] = pending_edges[index][int(token)]
                transitions[index] = np.asarray(
                    info["executed_transition_matrix"],
                    dtype=np.float64,
                )
                pending_edges[index] = np.asarray(
                    info["executed_edge_transition_matrices"],
                    dtype=np.float64,
                )
        first_call = False
        return {
            "source_operators": source_operators,
            "executed_transitions": transitions,
            "diagnostic_beliefs": np.stack(
                [info["belief_current"] for info in infos]
            ),
            "states": np.asarray(
                [info["state_current"] for info in infos],
                dtype=np.int64,
            ),
            "env_indices": np.arange(n_envs, dtype=np.int64),
            "episode_steps": np.asarray(episode_steps, dtype=np.int64),
            "episode_ids": episode_ids.copy(),
        }

    collected = collect_batched_rollout_data(
        env_factory,
        step_adapter,
        target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=n_envs,
        initial_state=initial_state,
        reset_state=reset_state,
        warmup=0,
    )
    targets = collected.targets
    groups = targets["episode_ids"]
    steps = targets["episode_steps"]
    beliefs = transducer_targets(
        initial_belief,
        targets["source_operators"],
        targets["executed_transitions"],
        groups,
        steps,
    )
    keep = steps >= warmup
    return ProbeData(
        activations=np.asarray(collected.representations, dtype=np.float64)[
            keep
        ],
        beliefs=beliefs[keep],
        diagnostic_beliefs=np.asarray(
            targets["diagnostic_beliefs"], dtype=np.float64
        )[keep],
        states=targets["states"][keep],
        actions=np.asarray(collected.actions, dtype=np.int64).reshape(-1)[
            keep
        ],
        rewards=np.asarray(collected.rewards, dtype=np.float64)[keep],
        env_indices=targets["env_indices"][keep],
        episode_steps=steps[keep],
        episode_ids=groups[keep],
    )


def probe_checkpoint(
    checkpoint: Path,
    *,
    preset: str,
    variant: int,
    reward_state: str,
    seed: int,
    train_steps: int,
    test_steps: int,
    warmup: int,
    device: str = "cpu",
    policy_mode: str = "policy",
) -> dict[str, Any]:
    module = load_module_only(checkpoint)
    factory = make_environment_factory(preset, variant, reward_state)
    streams = named_seed_sequences(seed, _STREAM_KEYS)
    common = {
        "device": device,
        "warmup": warmup,
        "policy_mode": policy_mode,
    }
    train = collect_probe_data(
        module,
        factory,
        n_steps=train_steps,
        seed=seed_sequence_to_int(streams["probe_train"], bits=32),
        **common,
    )
    test = collect_probe_data(
        module,
        factory,
        n_steps=test_steps,
        seed=seed_sequence_to_int(streams["probe_test"], bits=32),
        **common,
    )
    target_error = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "transducer target is misaligned with environment diagnostics: "
            f"{target_error:.3e}"
        )
    weight, bias = fit_affine_probe(
        train.activations,
        train.beliefs,
        ridge=PROBE_RIDGE,
    )
    predicted = probe_predict(weight, bias, test.activations)
    r2 = r2_score(predicted, test.beliefs)
    mse = float(np.mean(np.square(predicted - test.beliefs)))
    variance = float(np.mean(np.square(test.beliefs - test.beliefs.mean(0))))
    return {
        "checkpoint": str(checkpoint),
        "preset": preset,
        "variant": variant,
        "reward_state": reward_state,
        "target": "controlled_transducer_belief_delay_one_edge_emitting",
        "target_timing": "current-state belief at decision time (source "
        "posterior on tokens/actions, predicted through executed action)",
        "probe": "held_out_affine_least_squares",
        "probe_ridge": PROBE_RIDGE,
        "representation": "post_final_norm_residual_stream",
        "policy_mode": policy_mode,
        "n_envs": N_ENVS,
        "warmup": warmup,
        "n_fit": len(train.beliefs),
        "n_test": len(test.beliefs),
        "r_squared": float(r2),
        "one_minus_r_squared": float(1.0 - r2),
        "mse": mse,
        "target_variance": variance,
        "normalized_mse": mse / variance if variance > 0 else float("nan"),
        "target_consistency_max_abs": target_error,
        "test_reward_occupancy": float(test.rewards.mean()),
        "test_action_fractions": (
            np.bincount(test.actions, minlength=3) / len(test.actions)
        ).tolist(),
        "interpretation": (
            "Affine decodability does not establish causal policy use."
        ),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--variant", type=int, required=True)
    parser.add_argument("--reward-state", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--test-steps", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--policy-mode", default="policy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metrics = probe_checkpoint(
        args.checkpoint,
        preset=args.preset,
        variant=args.variant,
        reward_state=args.reward_state,
        seed=args.seed,
        train_steps=args.train_steps,
        test_steps=args.test_steps,
        warmup=args.warmup,
        device=args.device,
        policy_mode=args.policy_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _main()
