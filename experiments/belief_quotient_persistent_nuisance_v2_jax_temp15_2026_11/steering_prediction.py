"""Zero-safe nearest-simplex discounted references; not optimality certificates."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.environment import action_transitions, EMISSION_MATRIX
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import LAMBDAS, MODES
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_storage import require_artifacts


def target(b, mode):
    b = np.asarray(b, float)
    if b.shape[-1] != 3 or not np.isfinite(b).all():
        raise ValueError('expected finite three-coordinate beliefs')
    if mode == 'mirror':
        return b[..., [1, 0, 2]].copy()
    if mode == 'point':
        low = b[..., 0] < b[..., 1]
        return np.stack((np.where(low, 1-b[..., 2], 0), np.where(low, 0, 1-b[..., 2]), b[..., 2]), -1)
    if mode not in ('rho_flip', 'rho_point'):
        raise ValueError(mode)
    rho = 1-b[..., 2] if mode == 'rho_flip' else (b[..., 2] < .5).astype(float)
    mass = b[..., :2].sum(-1, keepdims=True)
    ratio = np.divide(b[..., :2], mass, out=np.full_like(b[..., :2], .5), where=mass > 0)
    return np.concatenate((ratio*(1-rho[..., None]), rho[..., None]), -1)


class Policy:
    def __init__(self, variant, n=100, gamma=.995, tolerance=1e-9, max_iterations=10000):
        self.transitions = action_transitions(variant)
        self.n, self.gamma, self.variant = n, gamma, variant
        self.grid = np.array([(i/n, j/n, 1-(i+j)/n) for i in range(n+1) for j in range(n+1-i)])
        self.lookup = np.full((n+1, n+1), -1, int)
        ij = np.rint(self.grid[:, :2]*n).astype(int)
        self.lookup[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
        prior = np.einsum('ks,asj->kaj', self.grid, self.transitions)
        weights = prior[:, :, None, :]*EMISSION_MATRIX.T[None, None]
        probability = weights.sum(-1)
        posterior = np.divide(weights, probability[..., None], out=np.full_like(weights, 1/3), where=probability[..., None] > 0)
        nxt = self.nearest(posterior)
        value = np.zeros(len(self.grid))
        reward = self.grid[:, 2]
        if variant == 2:
            rho = np.arange(n+1)/n
            r = (1-rho[:, None])*self.transitions[None, :, 0, 2]+rho[:, None]*self.transitions[None, :, 2, 2]
            probability = np.stack((.15*(1-r), r+.85*(1-r)), -1)
            posterior_rho = np.stack((np.zeros_like(r), r/probability[..., 1]), -1)
            nxt = np.rint(posterior_rho*n).astype(int)
            value, reward = np.zeros(n+1), rho
        for iteration in range(max_iterations):
            q = reward[:, None]+gamma*(probability*value[nxt]).sum(-1)
            new_value = q.max(-1)
            residual = float(np.max(np.abs(new_value-value)))
            value = new_value
            if residual < tolerance:
                break
        self.actions = q.argmax(-1)
        self.convergence = dict(iterations=iteration+1, bellman_residual=residual,
                                value_error_bound=residual/(1-gamma), converged=residual < tolerance,
                                n=n, gamma=gamma, tolerance=tolerance,
                                belief_grid='exact reward-state quotient' if variant == 2 else 'full simplex')

    def nearest(self, b):
        b = np.maximum(np.asarray(b), 0)
        b = b/b.sum(-1, keepdims=True)
        ij = np.rint(b[..., :2]*self.n).astype(int)
        i, j = ij[..., 0], ij[..., 1]
        over = i+j > self.n
        i, j = np.where(over & (b[..., 0] > b[..., 1]), self.n-j, i), np.where(over & (b[..., 0] <= b[..., 1]), self.n-i, j)
        return self.lookup[i, j]

    def __call__(self, b):
        if self.variant == 2:
            return self.actions[np.rint(np.clip(np.asarray(b)[..., 2], 0, 1)*self.n).astype(int)]
        return self.actions[self.nearest(b)]


def simulate(policy, mode, strength, episodes=32, length=1024, seed=2042, random=False):
    rng = np.random.default_rng(seed)
    states = rng.integers(3, size=episodes)
    b = np.full((episodes, 3), 1/3)
    occupied = np.zeros(episodes)
    actions = np.zeros(3)
    for _ in range(length):
        token = (rng.random(episodes)[:, None] >= EMISSION_MATRIX[states].cumsum(-1)).sum(-1)
        b *= EMISSION_MATRIX[:, token].T
        b /= b.sum(-1, keepdims=True)
        installed = (1-strength)*b+strength*target(b, mode)
        a = rng.integers(3, size=episodes) if random else policy(installed)
        occupied += states == 2
        actions += np.bincount(a, minlength=3)
        states = (rng.random(episodes)[:, None] >= policy.transitions[a, states].cumsum(-1)).sum(-1)
        b = np.einsum('bs,bsj->bj', b, policy.transitions[a])
    occupied /= length
    return dict(occupancy=float(occupied.mean()), occupancy_se=float(occupied.std(ddof=1)/np.sqrt(episodes)), action_fractions=(actions/(episodes*length)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    require_artifacts(args.output)
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError(args.output)
    results = {}
    for v in (1, 2, 3):
        for gamma in (.97, .99, .995):
            for n in (60, 100):
                policy = Policy(v, n=n, gamma=gamma)
                curves = {m: {str(l): simulate(policy, m, l, episodes=2 if args.smoke else 32, length=32 if args.smoke else 1024) for l in LAMBDAS} for m in MODES}
                results[f'v{v}_g{gamma}_n{n}'] = dict(convergence=policy.convergence, curves=curves,
                    intact=simulate(policy, 'mirror', 0, episodes=2 if args.smoke else 32, length=32 if args.smoke else 1024),
                    random=simulate(policy, 'mirror', 0, episodes=2 if args.smoke else 32, length=32 if args.smoke else 1024, random=True))
                print(v, gamma, n, policy.convergence, flush=True)
    sensitivity = {}
    for v in (1, 2, 3):
        for gamma in (.97, .99, .995):
            coarse = results[f'v{v}_g{gamma}_n60']
            fine = results[f'v{v}_g{gamma}_n100']
            delta = max(abs(coarse['curves'][m][str(l)]['occupancy']-fine['curves'][m][str(l)]['occupancy']) for m in MODES for l in LAMBDAS)
            sensitivity[f'v{v}_g{gamma}'] = dict(max_occupancy_grid_difference=delta,
                bellman_converged=coarse['convergence']['converged'] and fine['convergence']['converged'],
                grid_agreement_within_005=delta <= .005)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, shared_across_epsilon=True, seed=2042,
        episodes=2 if args.smoke else 32, episode_length=32 if args.smoke else 1024,
        headline={f'v{v}': dict(candidate='g0.995_n100',
            converged_at_tested_resolution=sensitivity[f'v{v}_g0.995']['bellman_converged'] and sensitivity[f'v{v}_g0.995']['grid_agreement_within_005']) for v in (1, 2, 3)},
        grid_sensitivity=sensitivity, results=results)
    args.output.write_text(json.dumps(report, indent=2))
    lines = ['# V2 numerical reference sensitivity', '',
             '| Variant | γ | N | Bellman residual | Converged | Intact occupancy |', '|---:|---:|---:|---:|---|---:|']
    for key, value in results.items():
        c = value['convergence']
        lines.append(f"| {key.split('_')[0]} | {c['gamma']} | {c['n']} | {c['bellman_residual']:.3g} | {c['converged']} | {value['intact']['occupancy']:.6f} |")
    lines += ['', 'Grid sensitivity (maximum occupancy difference across targets and λ, common random streams):', '', json.dumps(sensitivity, indent=2), '',
              'Full curves are in the companion JSON and apply to both epsilon studies. Bellman convergence alone does not demonstrate grid convergence.']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
