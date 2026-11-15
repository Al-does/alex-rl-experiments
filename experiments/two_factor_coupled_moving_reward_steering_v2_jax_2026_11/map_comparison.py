"""Next-step versus long-run action maps on the exact belief (no agent).

``steering.py`` scores both maps on the exact filtered belief with the 256
confirmation episodes, where their difference is inside the episode standard
error. This script repeats that comparison on many more episodes with common
random numbers, adds long-run maps at other discounts, and reports the reward
of the same maps when the hidden state is known.

Tables are built and cast exactly as in ``steering.evaluate``, so the 256-episode
check at the confirmation seed reproduces the per-agent reports.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import harness
from experiments.belief_quotient_history_jax_2026_11 import env as hmm
from experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11 import steering as s

base = s.base
STUDY_DIR = Path(__file__).resolve().parent
DISCOUNTS = (0.5, 0.9, s.GAMMA, 0.999)
NEXT_STEP, LONG_RUN = "next_step", f"long_run_gamma_{s.GAMMA}"


def maps(reward: str) -> dict[str, np.ndarray]:
    return {NEXT_STEP: s.q_tables(reward, "myopic"),
            **{f"long_run_gamma_{g}": s.q_tables(reward, "discounted", g) for g in DISCOUNTS}}


@partial(jax.jit, static_argnums=(1, 2))
def _rollout(process, episodes: int, known_state: bool, q, q_next_step, schedule, key):
    """Per-episode reward per step and fraction of steps departing from the next-step map.

    Keys are split as in ``steering._evaluate``, so equal seeds and episode
    counts give the same environment draws.
    """
    reset_key, step_key, _ = jax.random.split(key, 3)
    env_state = jax.vmap(hmm.reset, (None, 0))(process, jax.random.split(reset_key, episodes))
    belief = jax.vmap(hmm.belief_reset, (None, 0))(process, env_state.token)

    def body(state, t):
        env_state, belief, reward, differs = state
        action = jnp.argmax(q[t][env_state.state] if known_state else belief @ q[t], -1)
        reward = reward + schedule[t][env_state.state]
        differs = differs + (action != jnp.argmax(belief @ q_next_step[t], -1))
        next_env, _, _ = jax.vmap(hmm.step, (None, 0, 0, 0))(
            process, jax.random.split(jax.random.fold_in(step_key, t), episodes), env_state, action)
        belief = jax.vmap(hmm.belief_update, (None, 0, 0, 0))(process, belief, action, next_env.token)
        return (next_env, belief, reward, differs), None

    zeros = jnp.zeros(episodes)
    (_, _, reward, differs), _ = jax.lax.scan(
        body, (env_state, belief, zeros, zeros), jnp.arange(s.HORIZON))
    return reward / s.HORIZON, differs / s.HORIZON


def compare(reward: str, *, episodes: int, seed: int) -> dict:
    """Reward per step of every map on the exact belief and on the known state."""
    if episodes < 2:
        raise ValueError("at least two episodes are required")
    process = base.source_process("reward_both")
    schedule = jnp.asarray(s.reward_schedule(reward).sum(1), jnp.float32)
    tables = {name: jnp.asarray(q, jnp.float32) for name, q in maps(reward).items()}
    key = base._keys(seed)[0]

    def run(name, known_state):
        earned, differs = _rollout(process, episodes, known_state, tables[name],
                                   tables[NEXT_STEP], schedule, key)
        return np.asarray(earned, np.float64), float(np.mean(differs))

    def mean_se(values):
        return float(values.mean()), float(values.std(ddof=1) / math.sqrt(episodes))

    out, per_episode = {"exact_belief": {}, "known_state": {}}, {}
    for name in tables:
        per_episode[name], differs = run(name, False)
        mean, se = mean_se(per_episode[name])
        out["exact_belief"][name] = {"total_reward": mean, "total_reward_se": se,
                                     "action_differs_from_next_step_map": differs}
    for name in tables:
        difference, difference_se = mean_se(per_episode[name] - per_episode[NEXT_STEP])
        out["exact_belief"][name] |= {"paired_minus_next_step": difference,
                                      "paired_minus_next_step_se": difference_se}
    for name in (NEXT_STEP, LONG_RUN):
        mean, se = mean_se(run(name, True)[0])
        out["known_state"][name] = {"total_reward": mean, "total_reward_se": se}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=STUDY_DIR / "results" / "exact_belief_map_comparison.json")
    parser.add_argument("--episodes", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=3042)
    parser.add_argument("--check-episodes", type=int, default=256)
    parser.add_argument("--check-seed", type=int, default=2042)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; refusing to overwrite")
    report = {
        "revisions": {"experiment": base._git(STUDY_DIR),
                      "harness": base._git(Path(harness.__file__).resolve().parent)},
        "episode_horizon": s.HORIZON, "moving_reward_period": s.PERIOD,
        "episodes": args.episodes, "seed": args.seed,
        "paired": "all maps share the reset and transition keys of one seed",
        "rewards": {r: compare(r, episodes=args.episodes, seed=args.seed) for r in s.REWARDS},
        # Same keys and episode count as the steering confirmation: these reproduce
        # exact_belief_{myopic,discounted}_map.total_reward in the per-agent reports.
        "confirmation_check": {
            "episodes": args.check_episodes, "seed": args.check_seed,
            "rewards": {r: {k: v for k, v in compare(
                r, episodes=args.check_episodes, seed=args.check_seed)["exact_belief"].items()
                if k in (NEXT_STEP, LONG_RUN)} for r in s.REWARDS}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for reward, result in report["rewards"].items():
        for name, row in result["exact_belief"].items():
            print(f"{reward:7s} {name:22s} {row['total_reward']:.4f} ± {row['total_reward_se']:.4f}"
                  f"  vs next-step {row['paired_minus_next_step']:+.4f}"
                  f" ± {row['paired_minus_next_step_se']:.4f}"
                  f"  differs {row['action_differs_from_next_step_map']:.3f}")
        for name, row in result["known_state"].items():
            print(f"{reward:7s} known state, {name:22s} {row['total_reward']:.4f}")


if __name__ == "__main__":
    main()
