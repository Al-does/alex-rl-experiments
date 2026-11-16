"""Exact-filter references for steering in the epsilon-0.02 ladder.

A reference agent tracks the exact filter and acts greedily under a discounted
belief-grid policy on the installed belief ``target(b)``. The report gives the
arrival occupancy and the fraction of decisions that differ from the same
policy on the unmodified belief. Grid policies are numerical approximations,
not certified optima.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import MODES
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import Policy, target

OUTPUT = Path(__file__).with_name('results')/'belief_steering_reference.json'


def simulate(policy, write=None, episodes=256, length=1024, seed=2042, random=False):
    """``write`` maps the exact belief to the belief the reference agent acts on."""
    rng = np.random.default_rng(seed)
    states = rng.integers(3, size=episodes)
    b = np.full((episodes, 3), 1/3)
    occupied, changed = np.zeros(episodes), np.zeros(episodes)
    for _ in range(length):
        token = (rng.random(episodes)[:, None] >= policy.emission[states].cumsum(-1)).sum(-1)
        b = b*policy.emission[:, token].T
        b /= b.sum(-1, keepdims=True)
        intact = policy(b)
        action = rng.integers(3, size=episodes) if random else intact if write is None else policy(write(b))
        occupied += states == 2
        changed += action != intact
        states = (rng.random(episodes)[:, None] >= policy.transitions[action, states].cumsum(-1)).sum(-1)
        b = np.einsum('bs,bsj->bj', b, policy.transitions[action])
    occupied /= length
    return dict(occupancy=float(occupied.mean()), occupancy_se=float(occupied.std(ddof=1)/np.sqrt(episodes)),
                action_change=float(changed.mean()/length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--grid', type=int, default=100)
    args = parser.parse_args()
    report = dict(schema=1, emission=np.asarray(environment.EMISSION_MATRIX).tolist(), episodes=256, length=1024, variants={})
    for variant in (1, 2, 3):
        policy = Policy(variant, n=args.grid, gamma=.995, environment=environment)
        entry = dict(convergence=policy.convergence, intact=simulate(policy), random=simulate(policy, random=True))
        entry.update({mode: simulate(policy, lambda b, mode=mode: target(b, mode)) for mode in MODES})
        report['variants'][str(variant)] = entry
        print(variant, json.dumps({k: v for k, v in entry.items() if k != 'convergence'}), flush=True)
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
