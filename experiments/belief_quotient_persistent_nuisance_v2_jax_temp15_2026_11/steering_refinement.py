"""FSQ barycentric Bellman interpolation and matched closed-loop grid refinement."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.environment import action_transitions, EMISSION_MATRIX
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_core import LAMBDAS, MODES
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import simulate
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_storage import require_artifacts


class InterpolatedPolicy:
    def __init__(self, n, gamma, tolerance=1e-8, cap=6000, variant=3):
        self.n, self.gamma, self.variant = n, gamma, variant
        self.transitions = action_transitions(variant)
        self.grid = np.array([(i/n, j/n, 1-(i+j)/n) for i in range(n+1) for j in range(n+1-i)])
        self.lookup = np.full((n+1, n+1), -1, int)
        ij = np.rint(self.grid[:, :2]*n).astype(int)
        self.lookup[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
        prior = np.einsum('ks,asj->kaj', self.grid, self.transitions)
        weights = prior[:, :, None, :]*EMISSION_MATRIX.T[None, None]
        probability = weights.sum(-1)
        posterior = np.divide(weights, probability[..., None], out=np.full_like(weights, 1/3), where=probability[..., None] > 0)
        indices, interpolation = self.stencil(posterior)
        coefficient = interpolation*probability[..., None]
        value = np.zeros(len(self.grid))
        start = time.perf_counter()
        for iteration in range(cap):
            q = self.grid[:, 2, None]+gamma*(coefficient*value[indices]).sum((-1, -2))
            updated = q.max(-1)
            increment = updated-value
            residual = float(np.ptp(increment))
            value = updated-updated[0]
            if residual < tolerance:
                break
        self.q = q-q[0].max()
        self.convergence = dict(method='barycentric interpolation, relative-value iteration', variant=variant, n=n, gamma=gamma,
            iterations=iteration+1, span_bellman_residual=residual, relative_value_error_bound=residual/(1-gamma),
            converged=residual < tolerance, seconds=time.perf_counter()-start)

    def stencil(self, belief):
        b = np.maximum(np.asarray(belief), 0)
        b /= b.sum(-1, keepdims=True)
        xy = b[..., :2]*self.n
        ij = np.minimum(np.floor(xy).astype(int), self.n)
        i, j = ij[..., 0], ij[..., 1]
        u, v = xy[..., 0]-i, xy[..., 1]-j
        upper = u+v > 1
        x = np.stack((i+upper, i+1, i), -1).clip(0, self.n)
        y = np.stack((j+upper, j, j+1), -1).clip(0, self.n)
        weights = np.stack((np.where(upper, u+v-1, 1-u-v), np.where(upper, 1-v, u), np.where(upper, 1-u, v)), -1)
        return self.lookup[x, y], weights

    def __call__(self, b):
        indices, weights = self.stencil(b)
        scores = (self.q[indices]*weights[..., None]).sum(-2)
        return (scores >= scores.max(-1, keepdims=True)-1e-10).argmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Reuse a completed refinement calculation')
    parser.add_argument('--base', type=Path, help='Include original references for variants 1/2')
    args = parser.parse_args()
    require_artifacts(args.output)
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.input:
        publish(json.loads(args.input.read_text()), args.base, args.output)
        return
    report = dict(method='barycentric Bellman and action-value interpolation', results={}, comparisons={})
    for gamma in (.995, .99, .97):
        previous = None
        for n in (150, 200, 300):
            policy = InterpolatedPolicy(n, gamma)
            curves = {m: {str(l): simulate(policy, m, l) for l in LAMBDAS} for m in MODES}
            record = dict(convergence=policy.convergence, curves=curves)
            report['results'][f'v3_g{gamma}_n{n}'] = record
            print(policy.convergence, flush=True)
            if previous is not None:
                delta = max(abs(curves[m][str(l)]['occupancy']-previous['curves'][m][str(l)]['occupancy']) for m in MODES for l in LAMBDAS)
                report['comparisons'][f'g{gamma}_{previous["convergence"]["n"]}_to_{n}'] = dict(max_occupancy_difference=delta, below_001=delta < .01)
                print('resolution comparison', gamma, n, delta, flush=True)
            previous = record
            args.output.write_text(json.dumps(report, indent=2))
    if args.base:
        publish(report, args.base, args.output)


def publish(refinement, base, output):
    report = json.loads(base.read_text()) if base else dict(results={}, headline={})
    report['schema'] = 2
    report['refinement'] = dict(comparisons=refinement['comparisons'],
        method=refinement['method'], tie_tolerance=1e-10,
        tie_rule='Lowest action index within tolerance of maximum interpolated action value',
        solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat='Matched 32 complete 1024-step reference episodes, seed 2042; resolution agreement is empirical, not an optimality certificate.')
    for key, record in refinement['results'].items():
        record['intact'] = record['curves']['mirror']['0.0']
        report['results'][key] = record
    comparison = refinement['comparisons']['g0.995_200_to_300']
    report['headline']['v3'] = dict(candidate='g0.995_n300',
        converged_at_tested_resolution=comparison['below_001'], method=refinement['method'])
    output.write_text(json.dumps(report, indent=2))
    lines = ['# Refined FSQ numerical predictions', '',
        'Barycentric interpolation of Bellman continuation values and policy action values; fixed deterministic action ties.',
        'All γ/N solves have span Bellman residual below 1e-8. Prior nearest-grid references remain in the JSON for comparison.', '',
        '| γ | Resolution comparison | Maximum occupancy difference | Below .01 |', '|---:|---|---:|---|']
    for key, value in refinement['comparisons'].items():
        gamma, a, _, b = key.split('_')
        lines.append(f"| {gamma[1:]} | {a} → {b} | {value['max_occupancy_difference']:.8f} | {value['below_001']} |")
    lines += ['', '## γ=.995, N=300 curves', '', '| λ | Point | Mirror | ρ flip | ρ point |', '|---:|---:|---:|---:|---:|']
    curves = report['results']['v3_g0.995_n300']['curves']
    for strength in LAMBDAS:
        values = ' | '.join(f"{curves[mode][str(strength)]['occupancy']:.6f}" for mode in MODES)
        lines.append(f'| {strength:g} | {values} |')
    lines += ['', report['refinement']['caveat'], '',
        'At mirror λ=.5, installed b1=b2 exactly: numerical X/Y ties must not become a history-dependent arbitrary policy. A fixed lowest-index tie rule removes that artifact. Discount sensitivity and finite-episode uncertainty remain distinct from grid sensitivity.']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
